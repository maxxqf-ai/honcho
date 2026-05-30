# My Good Honcho

基于 [plastic-labs/honcho](https://github.com/plastic-labs/honcho) 的定制分支，用于中文个人 AI 助手的长期记忆系统。

上游 honcho 默认面向英文用户、使用 OpenAI 模型、1536 维嵌入。这个分支针对以下场景做了改造：

- 用国产模型（MiniMax-M2.7）替代 OpenAI，降低成本
- 用本地部署的嵌入模型替代云端嵌入，数据不出本机
- 全部 agent 提示词改为中文，让模型用中文生成 observation
- 修复了实际使用中发现的上游 bug 和设计缺陷

---

## 快速开始

```bash
git clone https://github.com/maxxqf-ai/honcho.git
cd honcho
git checkout mygoodhoncho
cp config.toml.example config.toml
# 编辑 config.toml，填入你的 MiniMax API key 和数据库连接
uv sync
uv run fastapi dev src/main.py
```

前置依赖：
- PostgreSQL 17（需安装 pgvector 扩展）
- Redis
- 本地嵌入服务（如 omlx，运行在 127.0.0.1:8000，模型 nomic-embed-text）
- MiniMax API key（https://platform.minimaxi.com）

---

## 所有修改及原因

### 1. 嵌入维度 1536 -> 768

**文件：** `src/config.py`, `src/models.py`

**改了什么：** VECTOR_DIMENSIONS 默认值从 1536 改为 768；MessageEmbedding 和 Document 的 Vector 列从 1536 维改为 768 维；注释掉了 pgvector 1536 维校验器。

**为什么：** 本地部署的 nomic-embed-text 输出 768 维向量，不是 OpenAI 的 1536 维。上游强制要求 pgvector 用 1536 维，本地部署需要绕过这个限制。768 维还有一个好处：向量索引更小、相似度搜索更快，个人使用场景下 768 维精度够用。

---

### 2. Stale Batch 超时兜底

**文件：** `src/config.py`, `src/deriver/queue_manager.py`

**改了什么：** 新增 STALE_BATCH_HOURS 配置（默认 5 小时）；queue_manager 在筛选待处理批次时，如果某个 work unit 的最早排队项超过这个时间，强制处理，不再等 token 阈值。

**为什么：** 上游 deriver 有批次合并机制——攒够一定 token 数才处理。但如果某个人很久没说话，最后一次消息产生的排队项可能永远凑不够阈值，就一直卡着不处理。结果就是 honcho_query 查不到最新消息的内容。5 小时超时确保即使没新消息触发，旧消息也不会无限等待。

---

### 3. Deriver 提示词中文化 + 分类标签

**文件：** `src/deriver/prompts.py`, `src/deriver/deriver.py`

**改了什么：**
- 整个 deriver 提示词从英文重写为中文
- 新增 [direct]/[ref]/[behavior] 三分类标签
- 加强了 JSON 输出格式约束（只允许 `{"explicit": [{"content": "..."}]}` 格式，禁止额外字段）
- 明确要求不要把外部引用、系统提示、粘贴内容当成用户自身事实
- deriver.py 中过滤只处理 peer_name == observed 的消息

**为什么：** 上游英文提示词有几个问题：
1. 模型经常把用户转述/粘贴的外部内容当成用户自己的事实（比如用户贴了一段 Python 文档，honcho 就记录"用户懂 Python"）
2. 模型有时输出非标准 JSON（带 tags、type 等额外字段），导致解析失败
3. 英文提示词 + 中文对话 = observation 质量差，模型经常用英文总结中文内容
4. 一个批次里多个 peer 的消息混在一起处理，会串主体

三分类标签的效果：
- [direct] = 用户自己说的事实，可信赖
- [ref] = 用户引用/转述的外部内容，不能当用户事实
- [behavior] = 用户的操作/指令，描述行为而非事实

---

### 4. Dialectic 提示词中文化

**文件：** `src/dialectic/prompts.py`

**改了什么：** dialectic agent 的系统提示词和工具说明全部改为中文。

**为什么：** dialectic 是 honcho 的查询接口（回答"这个用户喜欢什么"这类问题）。用中文提问时，英文提示词会让模型先翻译再思考，降低推理质量。改成中文后查询响应更准确，也避免了英文系统提示词和中文用户查询之间的语言冲突。

---

### 5. Dreamer 收敛检测 + 元梦过滤 + 去重

**文件：** `src/dreamer/orchestrator.py`, `src/dreamer/specialists.py`, `src/dreamer/dream_scheduler.py`

**改了什么：**
- 新增 DreamCollectionSnapshot：dream 前后各拍一次快照，对比 observation 数量和内容变化
- 新增 _is_meta_dream_observation：过滤 dreamer 自己产生的自我报告类 observation（包含"dreamer""queue""204""deriver""consolidation"等关键词的）
- 新增 _normalize_observation_content：对 observation 文本做归一化（去标点、小写、合并空格）用于粗粒度去重计数
- dream_scheduler 中查询 explicit count 时加了 deleted_at IS NULL 过滤
- 提示词全部中文化，并加了以下关键约束：
  - 语义重复清理是最高优先级任务
  - 不要创建"元 observation"（关于 dreamer 自身运行的总结）
  - 对人格、职业、长期习惯等高风险画像要更保守
  - 不要为了证明自己工作过而强行创建 observation

**为什么：** 上游 dreamer 有几个严重问题：
1. **元梦污染：** dreamer 会创建关于 dreamer 自己是否成功的 observation（"dreamer 成功完成了去重""队列为空"），这些 observation 又被下次 dream 读到，形成恶性循环。一段时间后记忆库里大量这种垃圾
2. **无收敛判断：** dreamer 不知道什么时候该停。即使已经没有有意义的 observation 可以合并了，它也会继续创建低质量的 observation 来"证明"自己做了一次 dream
3. **重复 counting 不准：** dream_scheduler 在计算当前 explicit 数量时没过滤已软删除的文档，导致阈值判断不准
4. **提示词不保守：** deduction/induction specialist 会根据少量对话就给人贴"理性务实""注重细节"这类人格标签，长期积累画像会失真
5. **不清理重复：** 同一个事实被 deriver 多次记录后，dreamer 没有主动清理语义重复的机制

---

### 6. Structured Output 严格模式

**文件：** `src/llm/backends/openai.py`, `src/utils/representation.py`

**改了什么：**
- OpenAI backend 的 JSON schema 加了 `"strict": true`
- ExplicitObservationBase 和 PromptRepresentation 两个 Pydantic model 加了 `extra="forbid"`

**为什么：** 模型有时在 structured output 里自作主张加额外字段（tags、type、direct 等），导致下游解析出错。`strict: true` 让 API 端强制约束输出格式，`extra="forbid"` 让 Pydantic 端拒绝额外字段，双重保险。

---

### 7. 其他

**文件：** `src/utils/agent_tools.py`, `src/reconciler/sync_vectors.py` 及对应测试文件

- agent_tools：工具定义和执行逻辑配合上述改动做了适配
- sync_vectors：向量同步逻辑增强，配合 768 维迁移
- 6 个测试文件更新以匹配上述改动

---

## 配置参考

config.toml.example 是本分支的完整配置模板，主要区别：

| 配置项 | 上游默认 | 本分支 |
|--------|----------|--------|
| 嵌入维度 | 1536 | 768 |
| 嵌入模型 | text-embedding-3-small | nomicai-embed-bf16 (本地) |
| LLM 模型 | gpt-5.4-mini | MiniMax-M2.7 |
| LLM 端点 | OpenAI | api.minimaxi.com |
| STALE_BATCH_HOURS | 无 | 5 |
| Deriver 轮询间隔 | 1s | 10s |
| Dreamer tool iterations | 20 | 5 |
| pgvector MIGRATED | false | true |

---

## 注意事项

- 这个分支基于上游某个 commit，不一定兼容最新上游。使用前确认你的 PostgreSQL 已安装 pgvector 扩展
- 嵌入维度一旦选定并写入数据，不能随意改。如果要从 1536 迁移到 768，需要清空向量列或重新生成所有嵌入
- MiniMax-M2.7 是性价比模型，能力和 GPT-4 级别有差距。如果对 observation 质量要求更高，可以把 model_config 里的模型换成更强的
- 本分支的提示词优化是针对中文场景的。如果你的主要使用语言是英文，上游原始版本可能更合适
