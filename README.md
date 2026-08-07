# Text-to-SQL with BART and GPT-2

This repository contains a Natural Language Processing project for generating SQL queries from natural-language questions and database schemas using two Transformer architectures:

1. Text-to-SQL generation using BART
2. Text-to-SQL generation using GPT-2

The project explores dataset preprocessing, sequence-to-sequence learning, causal language modeling, SQL generation, exact-match evaluation, SQL normalization, and error analysis using Hugging Face Transformers and PyTorch.

---

## Project Overview

### BART and GPT-2 for Text-to-SQL

The project focuses on translating natural-language questions into SQL queries based on a provided database schema.

Two Transformer architectures are compared:

* BART as an Encoder–Decoder model
* GPT-2 as a Decoder-Only model

The main stages include:

* Loading the Gretel Synthetic Text-to-SQL dataset
* Inspecting and standardizing dataset fields
* Creating training, development, and test subsets
* Constructing schema-aware model inputs
* Fine-tuning BART for sequence-to-sequence generation
* Fine-tuning GPT-2 for causal language modeling
* Masking GPT-2 prefix tokens during training
* Generating SQL queries on development and test data
* Calculating Raw Exact Match
* Calculating Normalized Exact Match
* Normalizing generated SQL queries
* Performing automatic error tagging
* Inspecting randomly selected model predictions
* Comparing Encoder–Decoder and Decoder-Only architectures

BART processes the question and schema through an encoder and generates SQL with a separate decoder, while GPT-2 represents the input context and target SQL within a single autoregressive sequence.

Notebook:

```text
text_to_sql_bart_gpt2/text_to_sql_bart_gpt2.ipynb
```

---

## Dataset

### Gretel Synthetic Text-to-SQL

The project uses the Gretel Synthetic Text-to-SQL dataset from Hugging Face:

```python
load_dataset("philschmid/gretel-synthetic-text-to-sql")
```

The original dataset fields are standardized to:

* `question` — natural-language request
* `schema` — database schema
* `query` — reference SQL query

The original training split is divided into training and development subsets using a fixed random state.

The experiments use the following sampled subsets:

* Training samples: **20,000**
* Development samples: **2,000**
* Test samples: **2,000**

The samples are selected with a fixed random state to improve reproducibility and reduce training time.

---

## BART Pipeline

### Input Construction

Each BART input combines the database schema and natural-language question using the following structure:

```text
Translate to SQL.
Schema:
<database schema>
Question:
<natural-language question>
SQL:
```

The reference SQL query is used as the target sequence.

### Tokenization

The BART preprocessing pipeline:

1. Builds the schema-aware input prompt.
2. Tokenizes the input sequence.
3. Truncates the input to a maximum length of 512 tokens.
4. Tokenizes the reference SQL query separately.
5. Truncates the target SQL to a maximum length of 256 tokens.
6. Stores the target token IDs as training labels.

### Model

The project uses:

```text
facebook/bart-base
```

The main training configuration includes:

* Batch size: `8`
* Gradient accumulation steps: `2`
* Learning rate: `5e-5`
* Training epochs: `2`
* Maximum source length: `512`
* Maximum target length: `256`
* Mixed-precision training when CUDA is available

The model is trained using Hugging Face `Seq2SeqTrainer`.

### SQL Generation

During evaluation:

1. The question and schema are formatted using the same BART prompt.
2. The input is tokenized.
3. SQL is generated using beam search.
4. The generated token sequence is decoded into SQL text.
5. The prediction is compared with the reference query.

---

## GPT-2 Pipeline

### Prefix Construction

GPT-2 receives the question and schema as a structured causal-language-modeling prefix:

```text
question: <natural-language question>
schema: <database schema>
SQL:
```

During training, the reference SQL query and end-of-sequence token are appended to this prefix.

### Label Masking

The GPT-2 preprocessing pipeline creates:

```text
Prefix + Reference SQL + EOS
```

The tokens belonging to the input prefix are replaced with `-100` in the training labels.

This prevents the prefix tokens from contributing to the loss and focuses optimization on generating the target SQL query.

### Dynamic Padding

A custom collate function is used to pad variable-length sequences within each batch.

The padding strategy uses:

* Tokenizer padding token for input IDs
* `0` for attention masks
* `-100` for labels

The `-100` label value ensures that padded positions are ignored during loss computation.

### Model

The project uses:

```text
gpt2
```

The main training configuration includes:

* Training batch size: `4`
* Evaluation batch size: `4`
* Gradient accumulation steps: `4`
* Learning rate: `5e-5`
* Training epochs: `2`
* Maximum sequence length: `768`
* Mixed-precision training when CUDA is available

The model is trained using Hugging Face `Trainer`.

### SQL Generation

During inference:

1. Only the question-schema prefix is provided to GPT-2.
2. The model generates the continuation autoregressively.
3. The input prefix tokens are removed from the generated sequence.
4. Only the generated SQL portion is decoded.
5. The prediction is compared with the reference SQL query.

The implementation supports deterministic beam search as well as sampling-based generation.

---

## Evaluation

Both models are evaluated using the same two metrics.

### Raw Exact Match

Raw Exact Match directly compares the predicted SQL string with the reference SQL string.

A prediction is considered correct only when the two strings are exactly identical.

### Normalized Exact Match

Normalized Exact Match reduces superficial SQL formatting differences before comparison.

The normalization pipeline:

1. Removes leading and trailing whitespace.
2. Removes trailing semicolons.
3. Converts SQL keywords to lowercase.
4. Removes SQL comments.
5. Standardizes spacing around operators.
6. Replaces repeated whitespace with a single space.

The normalized predicted and reference queries are then compared directly.

---

## Error Analysis

The project includes lightweight automatic error tagging for generated SQL queries.

The analysis checks for missing or additional SQL clauses including:

* `SELECT`
* `FROM`
* `WHERE`
* `GROUP BY`
* `ORDER BY`
* `JOIN`
* `LIMIT`

It also checks for mismatches involving aggregation functions:

* `COUNT`
* `AVG`
* `SUM`
* `MIN`
* `MAX`

Predictions that do not match any predefined error category are grouped under structural, column, or value mismatches.

In addition, five randomly selected development examples are inspected by displaying:

* Natural-language question
* Database schema
* Generated SQL query
* Reference SQL query
* Raw Exact Match
* Normalized Exact Match
* Error tags
* Normalized prediction
* Normalized reference query

---

## BART vs. GPT-2

The project compares two different approaches to conditional SQL generation.

### BART

BART uses an Encoder–Decoder architecture.

The question and schema are processed by the encoder, while the decoder generates the SQL query as a separate output sequence.

### GPT-2

GPT-2 uses a Decoder-Only architecture.

The question, schema, and SQL are represented within a single autoregressive sequence, with the input prefix masked from the training loss.

The comparison focuses on:

* Exact-match performance
* SQL generation behavior
* Schema utilization
* Structural SQL errors
* Aggregation errors
* Formatting differences
* Architectural differences between sequence-to-sequence and causal generation

---

## Sampling and Limitations

The experiments use sampled subsets instead of the complete available dataset to reduce GPU memory requirements and execution time.

The training configuration uses:

* **20,000 training samples**
* **2,000 development samples**
* **2,000 test samples**

As a result:

* The models are trained on only part of the available data.
* Performance may improve with larger training subsets.
* Exact Match does not determine whether two structurally different SQL queries are semantically equivalent.
* Normalized Exact Match handles formatting differences but does not provide execution-based equivalence.
* Generated queries may contain correct SQL logic while differing from the reference representation.
* Results may vary depending on model initialization, hardware, and generation settings.

The primary purpose of the project is to implement and compare complete Transformer-based Text-to-SQL pipelines rather than to achieve state-of-the-art benchmark performance.

---

## Repository Structure

```text
text-to-sql-with-transformers/
│
├── text_to_sql_bart_gpt2/
│   └── text_to_sql_bart_gpt2.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

Training outputs and model checkpoints are excluded from version control.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* BART
* GPT-2
* Pandas
* NumPy
* Scikit-learn
* sqlparse
* Regular Expressions
* Jupyter Notebook
* Google Colab

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Hamidreza-Talei/text-to-sql-with-transformers.git
cd text-to-sql-with-transformers
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

---

## Running the Notebook

Open:

```text
text_to_sql_bart_gpt2/text_to_sql_bart_gpt2.ipynb
```

The notebook automatically downloads the Gretel Synthetic Text-to-SQL dataset and the required pretrained Transformer models.

To verify reproducibility:

1. Restart the notebook kernel or runtime.
2. Select **Run All**.
3. Confirm that all cells execute in order.
4. Confirm that dataset preprocessing completes successfully.
5. Confirm that BART and GPT-2 training and evaluation execute sequentially.

GPU acceleration is recommended for model fine-tuning and SQL generation.

---

## Generated Files

Running the notebook may generate local training directories such as:

```text
bart_text2sql/
gpt2_text2sql/
```

These directories may contain temporary training outputs or model artifacts and should not be committed to the repository.

---

## Key Concepts

This repository demonstrates:

* Natural Language Processing
* Text-to-SQL generation
* Transformer architectures
* Encoder–Decoder modeling
* Decoder-Only modeling
* BART
* GPT-2
* Sequence-to-sequence learning
* Causal language modeling
* Prompt construction
* Schema-aware input formatting
* Tokenization
* Attention masks
* Label masking
* Dynamic padding
* Mixed-precision training
* Gradient accumulation
* Beam search
* Autoregressive generation
* SQL normalization
* Raw Exact Match
* Normalized Exact Match
* SQL error analysis
* Hugging Face Datasets
* Hugging Face Trainer
* PyTorch
