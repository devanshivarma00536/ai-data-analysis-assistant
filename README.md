# AI Data Analysis Assistant

Natural-language data analysis app — ask questions in plain English, get charts, Pandas/SQL code, insights, and PDF reports.

## Features

- Load **CSV, Excel, SQLite, MySQL**
- **Natural language queries** (OpenRouter LLM + rule-based fallback)
- Auto **data profiling** and cleaning
- Analysis: ranking, trends, correlation, forecasting, ABC, RFM, cohort, anomaly detection
- Auto **Plotly charts**
- **PDF report** export
- **Chat mode** for follow-up questions
- Sample datasets via `fetch_all_data.py` (not stored in Git)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY
python fetch_all_data.py   # download sample datasets into data/
streamlit run app.py
```

Open http://localhost:8501

## Project structure

```
app.py              Streamlit UI
src/                Core engine (loader, profiler, analyzer, charts, LLM)
data/               Sample datasets
fetch_all_data.py   Download open datasets
components/         UI components
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key from [openrouter.ai](https://openrouter.ai) |
| `OPENROUTER_MODEL` | Default: `openrouter/free` |

**Never commit `.env`** — it is listed in `.gitignore`.

## Example questions

- Top 10 name by market_cap_usd
- Show correlation between market cap and price
- Predict next 3 months trend
- Detect anomalies in temp_max_c
- Show summary statistics

## Tech stack

Python, Pandas, Streamlit, Plotly, SQLAlchemy, OpenRouter, fpdf2

## License

MIT (add your license if needed)
