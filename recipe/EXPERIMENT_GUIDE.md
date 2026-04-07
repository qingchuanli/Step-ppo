## HotpotQA 与 ALFWorld 实验运行指南

本指南只说明你在 **运行实验时需要做的步骤**，代码部分已经在 `recipe/hotpotqa` 和 `recipe/alfworld` 中实现完毕。

---

## 一、HotpotQA 向量检索 Agent 实验

### 1. 环境依赖

- 在项目根目录执行（如尚未安装）：

```bash
cd /Users/lqc/codes/PaperScout-main
pip3 install -U datasets pyarrow pandas qdrant-client FlagEmbedding
```

### 2. 准备 HotpotQA 数据（parquet + wiki 语料）

在项目根目录运行：

```bash
python3 recipe/hotpotqa/prepare_hotpotqa_arft.py \
  --output_dir data/corpus/hotpotqa
```

生成文件：

- `data/corpus/hotpotqa/train.parquet`
- `data/corpus/hotpotqa/validation.parquet`
- `data/corpus/hotpotqa/hpqa_corpus.jsonl`

### 3. 构建本地 Qdrant wiki 向量库

```bash
python3 recipe/hotpotqa/process_hotpotqa.py
```

这会在：

- `data/corpus/hotpotqa/qdrant_db/`

下建立 `hpqa_corpus` collection，供 `wiki_search` 工具使用。

### 4. 准备基础模型

设置基础模型路径（可用 HuggingFace 名称或本地路径）：

```bash
export HOTPOTQA_MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
```

### 5. 运行 HotpotQA 训练（step-level / token-level advantage）

两个示例脚本位于 `examples/`：

- **step-level advantage**（序列级聚合）：

```bash
bash examples/run_hotpotqa_step_adv.sh
```

- **token-level advantage**（token 级聚合）：

```bash
bash examples/run_hotpotqa_token_adv.sh
```

二者都已经自动：

- 使用 `data/corpus/hotpotqa/*.parquet`；
- 使用 `recipe/hotpotqa/base.yaml` 注册的 `hotpotqa_agent`；
- 关闭 reward model，只用 `recipe/hotpotqa/reward_fn.py` 中的 EM 规则奖励。

---

## 二、ALFWorld 结果奖励 Agent 实验

### 1. 安装 ALFWorld 及依赖

ALFWorld 需要单独安装（建议在同一环境中）：

```bash
cd /path/to/your/workspace
git clone https://github.com/alfworld/alfworld.git
pip install -e alfworld
```

> 如 ALFWorld API 有变动，需要根据实际版本微调 `recipe/alfworld/env/alfworld_wrapper.py` 中 `_make_alfworld_env` 和 `reset/step` 的调用方式。

### 2. 准备 ALFWorld 任务数据并转为 parquet

1. 使用 ALFWorld 官方脚本生成 train/val 任务列表（json 或 jsonl），保证每条包含：
   - `task_id`
   - `description` 或 `goal`
   - 可选：`difficulty`, `gamefile`, `room`, `seed` 等

2. 在本项目根目录运行转换脚本（路径示例需替换为你真实的任务文件路径）：

```bash
cd /Users/lqc/codes/PaperScout-main
python3 recipe/alfworld/prepare_alfworld_arft.py \
  --train_tasks /path/to/alfworld_train_tasks.jsonl \
  --val_tasks /path/to/alfworld_val_tasks.jsonl \
  --output_dir data/corpus/alfworld
```

生成文件：

- `data/corpus/alfworld/train.parquet`
- `data/corpus/alfworld/validation.parquet`

### 3. 准备 ALFWorld 训练用基础模型

设置模型路径（与 HotpotQA 可相同或不同）：

```bash
export ALFWORLD_MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
```

### 4. 运行 ALFWorld 训练（step-level / token-level advantage）

脚本位于 `examples/`：

- **step-level advantage**：

```bash
bash examples/run_alfworld_step_adv.sh
```

- **token-level advantage**：

```bash
bash examples/run_alfworld_token_adv.sh
```

这两个脚本会自动：

- 使用 `data/corpus/alfworld/*.parquet`；
- 使用 `recipe/alfworld/base.yaml` 中注册的 `alfworld_agent`（多步 `env_step` 工具驱动）；
- 关闭 reward model，只用 `recipe/alfworld/reward_fn.py` 中基于 success/fail 的 0/1 结果奖励。

---

## 三、Quick Debug 建议

若你想先进行小规模 sanity check，可以在命令行追加 override，例如：

```bash
... trainer.total_epochs=1 data.train_max_samples=100 data.val_max_samples=50
```

示例（HotpotQA step-level）：

```bash
bash examples/run_hotpotqa_step_adv.sh \
  trainer.total_epochs=1 \
  data.train_max_samples=200 \
  data.val_max_samples=50
```

示例（ALFWorld step-level）：

```bash
bash examples/run_alfworld_step_adv.sh \
  trainer.total_epochs=1 \
  data.train_max_samples=50 \
  data.val_max_samples=10
```

这样可以快速验证数据加载、AgentFlow、多步工具调用和规则奖励是否正常工作，再扩到完整训练规模。 

