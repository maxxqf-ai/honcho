"""
Dream cycle 的 agentic specialists。

每个 specialist 都是自治 agent：
1. 接收探索线索
2. 用工具搜索相关 observations
3. 创建新的 observations（deductive 或 inductive）
4. deduction specialist 还可以删重复或过时项
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import crud, schemas
from src.config import ConfiguredModelSettings, settings
from src.dependencies import tracked_db
from src.exceptions import ValidationException
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import DreamSpecialistEvent, emit
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import TokenTypes
from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    create_tool_executor,
)

logger = logging.getLogger(__name__)


def _require_specialist_model_config(
    model_config: ConfiguredModelSettings | None,
    *,
    specialist_name: str,
) -> ConfiguredModelSettings:
    if model_config is None:
        raise ValidationException(
            f"{specialist_name} MODEL_CONFIG must be resolved before use"
        )
    return model_config


@dataclass
class SpecialistResult:
    """Result of a specialist run for telemetry and aggregation."""

    run_id: str
    specialist_type: str
    iterations: int
    tool_calls_count: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    success: bool
    content: str


# Tool names to exclude when peer card creation is disabled
PEER_CARD_TOOL_NAMES = {"update_peer_card"}


class BaseSpecialist(ABC):
    """Base class for agentic specialists."""

    name: str = "base"
    # Subclasses can override to customize the peer card update instruction
    peer_card_update_instruction: str = (
        "只通过 `update_peer_card` 更新稳定、长期的人物卡事实。"
    )

    @abstractmethod
    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        """Get the tools available to this specialist."""
        ...

    @abstractmethod
    def get_model_config(self) -> ConfiguredModelSettings:
        """Get the configured model to use for this specialist."""
        ...

    def get_max_tokens(self) -> int:
        """Get max output tokens for this specialist."""
        return 16384

    def get_max_iterations(self) -> int:
        """Get max tool iterations."""
        return 15

    @abstractmethod
    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        """Build the system prompt for this specialist."""
        ...

    @abstractmethod
    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        """Build the user prompt with optional exploration hints and current peer card."""
        ...

    def _build_peer_card_context(self, peer_card: list[str] | None) -> str:
        """为 user prompt 构造人物卡上下文。"""
        if not peer_card:
            return ""
        facts = "\n".join(f"- {fact}" for fact in peer_card)
        return f"""
## 当前人物卡

{facts}

{self.peer_card_update_instruction}
如果要更新，提交完整、去重后的全量列表，并移除过时项。

"""

    async def run(
        self,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str | None,
        hints: list[str] | None = None,
        configuration: ResolvedConfiguration | None = None,
        parent_run_id: str | None = None,
    ) -> SpecialistResult:
        """
        Run the specialist agent.

        Uses short-lived DB sessions to avoid holding connections during LLM calls.

        Args:
            workspace_name: Workspace identifier
            observer: The observing peer
            observed: The peer being observed
            session_name: Session identifier
            hints: Optional hints to guide exploration (specialists explore freely if None)
            configuration: Resolved configuration for checking feature flags (optional)
            parent_run_id: Optional run_id from orchestrator for correlation

        Returns:
            SpecialistResult with metrics and content
        """
        run_id = parent_run_id or str(uuid.uuid4())[:8]
        task_name = f"dreamer_{self.name}_{run_id}"
        start_time = time.perf_counter()

        # Short-lived DB session for preflight operations
        async with tracked_db("dream.specialist.preflight") as db:
            await crud.get_peer(db, workspace_name, schemas.PeerCreate(name=observer))
            if observer != observed:
                await crud.get_peer(
                    db, workspace_name, schemas.PeerCreate(name=observed)
                )

            # Determine if peer card tools should be included
            peer_card_enabled = configuration is None or configuration.peer_card.create

            # Fetch current peer card to inject into prompt (saves a tool call)
            current_peer_card: list[str] | None = None
            if peer_card_enabled:
                current_peer_card = await crud.get_peer_card(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                )
        # DB session closed — LLM calls happen without holding a connection

        # Build messages
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    observed, peer_card_enabled=peer_card_enabled
                ),
            },
            {
                "role": "user",
                "content": self.build_user_prompt(hints, current_peer_card),
            },
        ]

        # Create tool executor with telemetry context
        tool_executor: Callable[
            [str, dict[str, Any]], Any
        ] = await create_tool_executor(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_observation_ids=True,
            history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
            configuration=configuration,
            run_id=run_id,
            agent_type=self.name,
            parent_category="dream",
        )

        model_config = self.get_model_config()

        # Respect operator-configured max_output_tokens on the specialist's
        # ModelConfig (e.g. DREAM_DEDUCTION_MODEL_CONFIG__MAX_OUTPUT_TOKENS).
        # Only fall back to the specialist's hardcoded default when the
        # config leaves max_output_tokens unset or non-positive.
        configured_max = model_config.max_output_tokens
        effective_max_tokens = (
            configured_max
            if configured_max and configured_max > 0
            else self.get_max_tokens()
        )

        # Track iterations via callback
        iteration_count = 0

        def iteration_callback(data: Any) -> None:
            nonlocal iteration_count
            iteration_count = data.iteration

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            model_config=model_config,
            prompt="",  # Ignored since we pass messages
            max_tokens=effective_max_tokens,
            tools=self.get_tools(peer_card_enabled=peer_card_enabled),
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=self.get_max_iterations(),
            messages=messages,
            track_name=f"Dreamer/{self.name}",
            iteration_callback=iteration_callback,
        )

        # Log metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", duration_ms, "ms")
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        accumulate_metric(task_name, "input_tokens", response.input_tokens, "count")
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "count")

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_dreamer_tokens(
                count=response.input_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.INPUT.value,
            )
            prometheus_metrics.record_dreamer_tokens(
                count=response.output_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.OUTPUT.value,
            )

        logger.info(
            f"{self.name}: Completed in {duration_ms:.0f}ms, "
            + f"{len(response.tool_calls_made)} tool calls, "
            + f"{response.input_tokens} in / {response.output_tokens} out"
        )

        log_performance_metrics(f"dreamer_{self.name}", run_id)

        # Emit telemetry event
        emit(
            DreamSpecialistEvent(
                run_id=run_id,
                specialist_type=self.name,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                iterations=iteration_count,
                tool_calls_count=len(response.tool_calls_made),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_ms=duration_ms,
                success=True,
            )
        )

        return SpecialistResult(
            run_id=run_id,
            specialist_type=self.name,
            iterations=iteration_count,
            tool_calls_count=len(response.tool_calls_made),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            success=True,
            content=response.content,
        )


class DeductionSpecialist(BaseSpecialist):
    """
    Creates deductive observations from explicit observations.

    This specialist:
    1. Explores recent observations and messages to understand what's there
    2. Identifies logical implications, knowledge updates, and contradictions
    3. Creates new deductive observations with premise linkage
    4. Deletes outdated observations
    5. Updates peer card with biographical facts
    """

    name: str = "deduction"
    peer_card_update_instruction: str = "只把稳定的人物资料/画像事实写入 `update_peer_card`。"

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return DEDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in DEDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model_config(self) -> ConfiguredModelSettings:
        return _require_specialist_model_config(
            settings.DREAM.DEDUCTION_MODEL_CONFIG,
            specialist_name="DREAM DEDUCTION",
        )

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 12

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = """

## 人物卡（必须遵守）

人物卡是稳定人物事实的摘要。出现以下信息时，必须考虑更新：
- 姓名、年龄、地点、职业
- 家庭成员与关系
- 长期有效的指令（例如“叫我 X”“不要提 Y”）
- 核心偏好与稳定特征

不要加入临时事件总结、一次性结论、推理痕迹、矛盾记录。

格式：
- 普通事实：`姓名：Alice`、`在 Google 工作`、`住在 NYC`
- 长期指令：`INSTRUCTION: ...`
- 偏好：`PREFERENCE: ...`
- 特征：`TRAIT: ...`

当你获得新的稳定人物信息时，用 `update_peer_card` 提交完整更新后的列表。
保持简洁、去重、最新，最多 40 条。"""

        return f"""你是一个 deductive reasoning agent，负责分析关于 {observed} 的 observations。

默认用简体中文生成 observation、premises、sources、peer card 内容。
命令名、标签名、工具名、模型名、路径、URL、版本号可保留原文。

## 你的任务

通过已有信息之间的逻辑关系，创建 deductive observations。像侦探一样连证据，不要脑补。

## 阶段 1：发现

先弄清记忆里到底有什么。可自由使用这些工具：
- `get_recent_observations`：看最近学到的内容
- `search_memory`：按主题搜索 observations
- `search_messages`：查看真实对话内容

先花几次工具调用理解材料，再决定是否创建新 observation。

## 阶段 2：动作

理解现状后，再创建 observation 并清理旧项：

### 语义重复清理（最高优先级）
当多条 observation 其实在表达同一事实，只是措辞不同、细节多少不同或重复提炼：
- 先找出重复组
- 保留信息最完整、措辞最稳定、source 更扎实的一条
- 对其余重复项调用 `delete_observations`

这类清理本身就是有效产出。
如果本轮只发现重复、没有可靠的新 deduction，也可以只删除，不必强行新建 observation。
不要为了证明自己工作过而创建“这里有重复”“模型能力不足”“队列为空/不为空”“dreamer 是否成功”这类元 observation。
如果 observation 里含有事件细节，例如具体日期、时间、路线、方向、出发/到达、爬山路径、交通方式：
- 默认保留这些 explicit observation
- 不要用抽象总结替代它们
- 只有当另一条 explicit observation 明确保留了同样细节时，才允许删除旧条目

### 知识更新（高优先级）
当同一事实在不同时间有不同值：
- `周二开会` [旧] → `会议改到周四` [新]
- 创建一条 deductive update observation
- 立刻删除过时 observation

### 逻辑蕴含
提取明确支持、且高置信度成立的隐含信息：
- `在 Google 做 SWE` → `具备软件工程技能`、`在科技行业任职`
- `孩子 5 岁和 8 岁` → `是家长`、`有学龄儿童`

### 矛盾
当两条陈述不能同时为真（不是单纯更新）时，标出来：
- `我喜欢咖啡` vs `我讨厌咖啡` → contradiction observation
{peer_card_section}

## 创建 observation

使用 `create_observations_deductive`。

```json
{{
  "observations": [{{
    "content": "逻辑结论",
    "source_ids": ["id1", "id2"],
    "premises": ["前提 1", "前提 2"]
  }}]
}}
```

## 规则

1. 不要长篇解释推理过程；主要靠工具和 observation 落地
2. 只基于你**实际找到**的证据创建 observation，不要按预期脑补
3. 一定要带 `source_ids`，指向你综合的来源 observation
4. `source_ids` 为空或缺失会被拒绝
5. 发现过时 observation 或语义重复 observation 就删，不要留重复
6. 质量优先于数量；少量强 deduction 胜过大量弱 deduction
7. 对职业、人格、长期习惯、家庭、地理位置等敏感画像要更保守，除非证据非常直接或逻辑必然
8. 先做 discovery，再决定创建还是删除；不要一上来就写 deduction
9. 输出优先使用简体中文"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}先从最近的 observations 和 messages 开始探索。以下主题可能值得调查：

{hints_str}

但以证据为准。如果你发现更重要、更明确的线索，就跟过去。

先用 `get_recent_observations` 看现状。"""

        return f"""{peer_card_context}探索 observation 空间，创建 deductive observations。

先用 `get_recent_observations` 看最近新增内容，再深入最有价值的线索。

重点看：
1. 语义重复（同一事实被多次记录）
2. 知识更新（同一事实随时间出现不同值）
3. 尚未显式写出的逻辑蕴含
4. 需要标记的矛盾

开始。"""


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Explores observations to understand what's there
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage
    4. Updates peer card with high-confidence traits and tendencies
    """

    name: str = "induction"
    peer_card_update_instruction: str = "只添加高度稳定的人物特征/偏好；不要复制瞬时结论。"

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return INDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in INDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model_config(self) -> ConfiguredModelSettings:
        return _require_specialist_model_config(
            settings.DREAM.INDUCTION_MODEL_CONFIG,
            specialist_name="DREAM INDUCTION",
        )

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 10

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = """

## 人物卡（必须遵守）

识别出模式后，只有在确实属于长期稳定画像时，才更新人物卡：
- `TRAIT: 分析型思维`
- `PREFERENCE: 偏好详细解释`

不要加入临时模式、单次事件结论、推理摘要。
只有在真的需要更新长期画像时，才用 `update_peer_card` 提交完整去重列表。
保持简洁，最多 40 条。"""

        return f"""你是一个 inductive reasoning agent，负责识别关于 {observed} 的模式。

默认用简体中文生成 observation、sources、peer card 内容。
命令名、标签名、工具名、模型名、路径、URL、版本号可保留原文。

## 你的任务

通过多个 observations 之间的共性，创建 inductive observations。要像谨慎的行为分析者，而不是随意贴人格标签。

## 阶段 1：发现

广泛探索，寻找模式。可用这些工具：
- `get_recent_observations`：最近新增内容
- `search_memory`：按主题搜索
- `search_messages`：查看真实对话

要同时看 explicit observations 和 deductive observations。很多模式需要跨层综合才能看出来。

## 阶段 2：动作

只有在证据支持时，才创建 inductive observations：

### 行为模式
- `在压力下更容易改期`
- `做决定前会先征求意见`
- `项目过程常呈现固定阶段`

### 偏好
- `偏好上午开会`
- `喜欢详细的技术解释`

### 性格/特征
- `整体更偏乐观`
- `规划时较重细节`

### 时间模式
- `职业目标长期保持一致`
- `居住情况变动较频繁`
{peer_card_section}

## 创建 observation

使用 `create_observations_inductive`。

```json
{{
  "observations": [{{
    "content": "模式或归纳结论",
    "source_ids": ["id1", "id2", "id3"],
    "sources": ["证据 1", "证据 2"],
    "pattern_type": "tendency",  // preference|behavior|personality|tendency|correlation
    "confidence": "medium"  // low=2条来源, medium=3-4条, high=5条及以上
  }}]
}}
```

## 规则

1. 至少需要 2 条 source observations；模式必须有证据
2. 不要把单条事实换个说法就当模式
3. `confidence` 按证据数量给：2=low，3-4=medium，5+=high
4. 多观察“随时间如何变化”，不要只看静态事实
5. 必须带 `source_ids`，始终可回溯到证据
6. `source_ids` 为空或缺失会被拒绝
7. 对人格、长期习惯、稳定偏好这类高风险结论要保守；如果证据弱，就不要写
8. 不要创建关于 dreamer、queue、204、deriver、schedule_dream、系统是否成功去重 这类自我总结 observation
9. 输出优先使用简体中文"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}开始探索并寻找模式。以下方向可能值得调查：

{hints_str}

但仍以证据为准；如果别处出现更扎实的模式，就跟过去。

先用 `get_recent_observations`。"""

        return f"""{peer_card_context}探索 observation 空间，识别模式。

记住：模式至少需要 2 条来源。重点找倾向、偏好、行为规律。

开始。"""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
