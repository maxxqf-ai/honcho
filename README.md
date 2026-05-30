# My Good Honcho

基于 [plastic-labs/honcho](https://github.com/plastic-labs/honcho) 的定制分支——面向中文个人 AI 助手的长期记忆系统。

上游 honcho 默认面向英文用户、使用 OpenAI 模型、1536 维嵌入。这个分支针对中文场景做了完整改造：国产模型替代 OpenAI、本地嵌入替代云端、全部 agent 提示词中文化、修复实际使用中发现的上游 bug 和设计缺陷。

## 快速开始

```bash
git clone https://github.com/maxxqf-ai/honcho.git
cd honcho
cp config.toml.example config.toml
# 编辑 config.toml，填入你的 MiniMax API key 和数据库连接
uv sync
uv run fastapi dev src/main.py
```

前置依赖：
- PostgreSQL 17 + pgvector 扩展
- Redis
- 本地嵌入服务（如 omlx，127.0.0.1:8000，模型 nomic-embed-text）
- MiniMax API key（https://platform.minimaxi.com）

## 配置对比

| 配置项           | 上游默认              | 本分支                  |
|------------------|-----------------------|-------------------------|
| 嵌入维度         | 1536                  | 768                     |
| 嵌入模型         | text-embedding-3-small | nomicai-embed-bf16 (本地) |
| LLM 模型         | gpt-5.4-mini          | MiniMax-M2.7            |
| LLM 端点         | OpenAI                | api.minimaxi.com        |
| STALE_BATCH_HOURS | 无                   | 5                       |
| Deriver 轮询间隔  | 1s                    | 10s                     |
| Dreamer iterations | 20                   | 5                       |

## 所有修改清单

详细的修改说明、原因和涉及的文件，请看 [MYGOODHONCHO.md](./MYGOODHONCHO.md)。

简要列表：

1. **嵌入维度 1536 -> 768** — 适配本地 nomic-embed-text，向量更小更快
2. **Stale Batch 超时兜底** — 防止消息永远凑不够阈值导致不处理（STALE_BATCH_HOURS=5）
3. **Deriver 提示词中文化 + 三分类标签** — [direct]/[ref]/[behavior] 区分用户事实和外部引用
4. **Dialectic 提示词中文化** — 中文查询不再经过英文提示词中转
5. **Dreamer 收敛检测 + 元梦过滤 + 去重** — 防止 dreamer 产生自我总结垃圾、重复 observation、乱贴人格标签
6. **Structured Output 严格模式** — strict=true + extra=forbid 双重保险防模型输出额外字段
7. **其他** — agent_tools 适配、sync_vectors 增强、测试更新

## 注意事项

- 嵌入维度选定后不能随意改，需要清空向量列或重新生成所有嵌入
- MiniMax-M2.7 是性价比模型，如需更高质量可替换为更强模型
- 提示词优化针对中文场景，主要语言为英文的建议用上游原版
- 基于 upstream 某个 commit，不保证兼容最新上游

## 致谢

- [plastic-labs/honcho](https://github.com/plastic-labs/honcho) — 原始项目
- [MiniMax](https://platform.minimaxi.com) — LLM API
- [nomic-ai/nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) — 嵌入模型
