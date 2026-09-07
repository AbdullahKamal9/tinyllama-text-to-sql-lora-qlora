# TinyLlama Text-to-SQL: LoRA vs QLoRA

Fine-tuned [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) to generate SQL queries from natural-language questions and a database schema, comparing two parameter-efficient fine-tuning approaches — **LoRA** and **QLoRA** — on the same task, same data, and same adapter configuration.

## What it does

Given a table schema and a plain-English question, the model generates a SQL query to answer it.

**Input:**
```
Schema: CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)
Question: What is the average salary in the Engineering department?
```

**Output:**
```sql
SELECT AVG(salary) FROM employees WHERE department = 'Engineering'
```

## Why LoRA vs QLoRA

Rather than fine-tuning with just one method, this project trains the same base model, the same LoRA rank/config, and the same dataset under two conditions:

- **QLoRA** — base model loaded in 4-bit (NF4 quantization), LoRA adapters trained on top. Lower memory footprint.
- **LoRA** — base model loaded in fp16 (no quantization), LoRA adapters trained on top. Higher memory footprint, no quantization error.

The goal is a direct, apples-to-apples comparison of training time, peak GPU memory, and downstream accuracy — not just "I ran a fine-tune."

## Dataset

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — natural language question + `CREATE TABLE` schema pairs mapped to the corresponding SQL query.

## Evaluation

Standard training loss isn't a meaningful result on its own for this task, so evaluation uses **execution accuracy**: the model's generated SQL and the gold SQL are both run against an in-memory SQLite database built from the example's schema, and a prediction counts as correct only if it returns the same result set as the gold query — not just similar-looking text.

## Results

| Run   | Train time | Peak VRAM | Execution accuracy |
|-------|-----------|-----------|---------------------|
| QLoRA | _TBD_     | _TBD_     | _TBD_               |
| LoRA  | _TBD_     | _TBD_     | _TBD_               |

*(Fill in after both training runs complete — see `finetune_sql_lora_and_qlora.py`, which prints this table automatically at the end of the run.)*

## Setup

```bash
pip install -r requirements.txt
```

Trained and tested on Google Colab's free-tier T4 GPU (16GB VRAM).

## Usage

```bash
python finetune_sql_lora_and_qlora.py
```

Runs both the QLoRA and LoRA experiments sequentially, freeing GPU memory between runs, and saves each adapter to `tinyllama-sql-runs/<qlora|lora>/adapter/`.

To try the fine-tuned model on your own schema and question after training:

```python
my_model = load_finetuned_model(all_results[0]["adapter_path"], use_qlora=True)

schema = "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)"
question = "What is the average salary in the Engineering department?"

sql = ask(question, schema, my_model)
print(sql)
```

## Project structure

```
.
├── finetune_sql_lora_and_qlora.py   # training, eval, and inference code
├── requirements.txt
├── tinyllama-sql-runs/
│   ├── qlora/adapter/                # QLoRA adapter weights
│   └── lora/adapter/                 # LoRA adapter weights
└── README.md
```

## Limitations

- Trained on a subset of the full dataset (configurable via `TRAIN_SUBSET_SIZE`) to fit within free-tier Colab session limits — not the full ~78k examples.
- Execution-accuracy evaluation builds tables from schema only, with no seeded data — it checks whether the query *runs* and returns a matching (empty) result structure, not correctness against populated rows.
- TinyLlama-1.1B is a small base model; accuracy is expected to be lower than what a larger base model would achieve on the same task.

## License

MIT
