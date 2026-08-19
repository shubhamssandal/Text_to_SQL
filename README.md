
# Text-to-SQL Chatbot 🗄️🤖

Convert plain-English questions into MySQL queries using LLMs — with an automated evaluation layer to score query quality, faithfulness, and safety.

## Overview

This project is an end-to-end **Text-to-SQL pipeline** built with [LangChain](https://github.com/langchain-ai/langchain). It connects to a live MySQL database, reads the schema automatically, and uses an LLM to translate natural-language questions into executable SQL — no manual query writing required.

On top of query generation, the project includes a full **evaluation harness** powered by [Ragas](https://github.com/explodinggradients/ragas), scoring every generated query on:
- **Context Precision** — how well the query aligns with the retrieved schema
- **Faithfulness** — whether the query is grounded in the actual schema (no hallucinated columns/tables)
- **Maliciousness** (Aspect Critic) — a safety check to flag harmful or deceptive queries
- **Helpfulness** (Rubric-based scoring, 1–5) — how useful and complete the generated query is

## Features

- 🔌 **Live schema introspection** — automatically reads table/column definitions from your MySQL database, so prompts always stay in sync with the actual schema
- 🧠 **Pluggable LLM backends** — swap between **Groq** (`gemma2-9b-it`, `openai/gpt-oss-20b`), **Google Gemini**, and **Vertex AI** with a one-line change
- 📊 **Automated SQL evaluation** — benchmark generated queries against reference SQL using Ragas metrics
- 🧩 **LangChain LCEL pipeline** — clean, composable chain: `schema → prompt → LLM → SQL string`
- 🔐 **Environment-based config** — credentials and API keys are kept out of source control via `.env`

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain / LangChain Community |
| LLMs | Groq, Google Gemini (via `langchain-google-genai`), Vertex AI |
| Database | MySQL (via `SQLAlchemy` + `PyMySQL`) |
| Embeddings | HuggingFace `sentence-transformers/all-mpnet-base-v2` |
| Evaluation | Ragas |
| Environment | Python 3.12, `python-dotenv` |

## Project Structure

```
Text-to-SQL-Chatbot/
├── text_to_sql.ipynb      # Main notebook: schema loading, SQL chain, evaluation
├── .env                   # Environment variables (not committed)
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/pik1989/Text-to-SQL-Chatbot.git
cd Text-to-SQL-Chatbot
```

### 2. Create a virtual environment
```bash
conda create -n text_to_sql python=3.12
conda activate text_to_sql
```

### 3. Install dependencies
```bash
pip install langchain langchain-community langchain-openai langchain-groq \
            langchain-google-genai langchain-huggingface langchain-core \
            pymysql sentence-transformers ragas python-dotenv
```

### 4. Configure environment variables
Create a `.env` file in the project root:

```env
# MySQL connection
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=text_to_sql

# LLM providers
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 5. Load your database
Import your dataset into MySQL under the database name set in `MYSQL_DATABASE`. The pipeline works with any schema — it introspects tables automatically.

## Usage

Run the notebook cells in order, or adapt the core chain into a script:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Using this MySQL table schema:
{schema}

Write a SQL query for this question:
{question}

Return only the SQL query on one line. Do not use Markdown code fences.
""")

sql_chain = (
    RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
    | prompt
    | llm
    | StrOutputParser()
)

query = sql_chain.invoke({"question": "What was the budget of Product 12?"})
print(query)
# SELECT `2017 Budgets` FROM `2017_budgets` WHERE `Product Name` = 'Product 12';
```

## Evaluation

The project scores generated SQL against reference queries using Ragas:

```python
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextPrecision, AspectCritic, RubricsScore

samples = [
    SingleTurnSample(
        user_input=question,
        response=generated_sql,
        retrieved_contexts=[schema_context],
        reference=reference_sql,
    )
    for question, generated_sql, reference_sql in zip(questions, generated, references)
]

results = evaluate(
    dataset=EvaluationDataset(samples=samples),
    metrics=[context_precision, aspect_critic, rubrics_score],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)
```

This produces a scored dataframe (Context Precision, Maliciousness, Helpfulness) for every question–query pair, useful for regression-testing prompt or model changes.

## Sample Results

| Question | Generated SQL | Context Precision | Helpfulness |
|---|---|---|---|
| What was the budget of Product 12? | `SELECT \`2017 Budgets\` FROM \`2017_budgets\` WHERE \`Product Name\` = 'Product 12';` | 1.0 | 4/5 |
| List all customer names | `SELECT \`Customer Names\` FROM customers;` | 1.0 | 5/5 |
| Find name and state of all regions | `SELECT name, state FROM regions;` | 1.0 | 5/5 |

## Roadmap

- [ ] Add query execution + result validation (not just SQL generation)
- [ ] Support additional databases (PostgreSQL, SQLite)
- [ ] Build a lightweight Streamlit UI for interactive querying
- [ ] Expand evaluation set with edge-case and multi-table join questions

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ragas](https://github.com/explodinggradients/ragas)
- [Groq](https://groq.com/)
