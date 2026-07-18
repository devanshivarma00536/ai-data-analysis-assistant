"""Streamlit app — AI Data Analysis Assistant."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzer import result_to_dataframe, run_analysis
from src.charts import create_plotly_chart, select_chart
from src.cleaner import clean_dataframe
from src.config import get_openrouter_model
from src.formatter import format_analysis_report
from src.intent_parser import parse_intent
from src.llm_client import is_llm_configured, parse_question_with_llm
from src.loader import list_data_files, load_file, load_mysql, load_sqlite
from src.profiler import profile_dataframe
from src.report_export import build_pdf_report

st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""


def dataset_examples(profile: dict) -> list[str]:
    """Build example questions that match the loaded dataset columns."""
    numeric = profile["roles"].get("numeric", [])
    categorical = profile["roles"].get("categorical", []) + profile["roles"].get("text", [])
    datetime_cols = profile["roles"].get("datetime", [])

    examples = ["Show summary statistics"]
    if numeric and categorical:
        examples.append(f"Top 10 {categorical[0]} by {numeric[0]}")
    if len(numeric) >= 2:
        examples.append(f"Show correlation between {numeric[0]} and {numeric[1]}")
    if numeric and datetime_cols:
        examples.append(f"Show trend over time for {numeric[0]}")
        examples.append(f"Predict next 3 months for {numeric[0]}")
    if numeric and categorical:
        examples.append(f"Run ABC analysis on {categorical[0]} by {numeric[0]}")
    if numeric:
        examples.append(f"Detect anomalies in {numeric[0]}")
    return examples


@st.cache_data(show_spinner=False)
def load_dataset_cached(source_key: str, path: str) -> pd.DataFrame:
    return load_file(path)


def load_uploaded_file(uploaded) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded)
    if suffix in {".sqlite", ".db"}:
        temp = ROOT / "data" / "_uploaded_temp.db"
        temp.write_bytes(uploaded.getvalue())
        return load_sqlite(temp)
    raise ValueError(f"Unsupported upload type: {suffix}")


def run_query(
    question: str,
    df: pd.DataFrame,
    profile: dict,
    use_llm: bool,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[dict, str, str | None]:
    parser_mode = "Rule-based"
    llm_error = None
    if use_llm and is_llm_configured():
        intent, llm_error = parse_question_with_llm(question, profile, df, chat_history)
        if intent:
            parser_mode = "OpenRouter AI"
        else:
            intent = parse_intent(question, profile["roles"], profile["column_names"])
    else:
        intent = parse_intent(question, profile["roles"], profile["column_names"])

    raw_result, pandas_code, sql_code = run_analysis(df, intent)
    result_df = result_to_dataframe(raw_result)
    chart_type, chart_reason = select_chart(intent, result_df)
    report = format_analysis_report(
        question=question,
        intent=intent,
        pandas_code=pandas_code,
        sql_code=sql_code,
        result_df=result_df,
        chart_type=chart_type,
        chart_reason=chart_reason,
        profile=profile,
    )
    report["parser_mode"] = parser_mode
    report["llm_error"] = llm_error
    report["result_df"] = result_df
    report["intent_obj"] = intent
    return report, parser_mode, llm_error


def render_report(report: dict) -> None:
    st.markdown("---")
    st.subheader("User Question")
    st.write(report["user_question"])

    st.subheader("Detected Intent")
    st.write(report["detected_intent"])
    st.caption(report["intent_detail"])

    st.subheader("Relevant Columns")
    st.write(", ".join(report["relevant_columns"]) if report["relevant_columns"] else "—")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generated Pandas Code")
        st.code(report["pandas_code"], language="python")
    with col2:
        st.subheader("Generated SQL (read-only)")
        st.code(report["sql_code"], language="sql")

    st.subheader("Visualization")
    st.write(f"**Chart Type:** {report['chart_type']}")
    st.write(f"**Reason:** {report['chart_reason']}")

    st.subheader("Analysis")
    for item in report["analysis"]:
        st.write(f"- {item}")

    st.subheader("Business Insights")
    for item in report["business_insights"]:
        st.write(f"- {item}")

    st.subheader("Recommendations")
    for item in report["recommendations"]:
        st.write(f"- {item}")

    st.subheader("Confidence Score")
    st.progress(report["confidence_score"] / 100)
    st.write(f"**{report['confidence_score']}%**")


def show_chart(report: dict) -> None:
    intent = report["intent_obj"]
    result_df = report["result_df"]
    fig = create_plotly_chart(intent, result_df, report["chart_type"])
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(result_df, use_container_width=True)


st.title("AI Data Analysis Assistant")
st.caption("Upload data, ask questions by text or voice, export PDF reports.")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose source", ["Sample datasets", "Upload file", "MySQL"])

    df: pd.DataFrame | None = None
    source_label = ""

    if source == "Sample datasets":
        files = list_data_files(ROOT / "data")
        if not files:
            st.warning("No sample data found. Run: python fetch_all_data.py")
        else:
            labels = [f.name for f in files]
            pick = st.selectbox("Dataset", labels)
            path = str(ROOT / "data" / pick)
            df = load_dataset_cached(pick, path)
            source_label = pick

    elif source == "Upload file":
        uploaded = st.file_uploader("CSV, Excel, or SQLite", type=["csv", "xlsx", "xls", "sqlite", "db"])
        if uploaded:
            df = load_uploaded_file(uploaded)
            source_label = uploaded.name

    else:
        st.caption("MySQL connection (read-only SELECT)")
        host = st.text_input("Host", "localhost")
        port = st.number_input("Port", value=3306, step=1)
        user = st.text_input("User", "root")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database")
        table = st.text_input("Table (optional)")
        if st.button("Connect & Load"):
            try:
                df = load_mysql(
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=int(port),
                    table=table or None,
                )
                source_label = f"mysql:{database}.{table or 'auto'}"
            except Exception as exc:
                st.error(str(exc))

    clean_toggle = st.checkbox("Auto-clean data", value=True)

    st.divider()
    st.header("AI Settings")
    llm_available = is_llm_configured()
    use_llm = st.checkbox("Use OpenRouter AI", value=llm_available, disabled=not llm_available)
    if llm_available:
        st.caption(f"Model: `{get_openrouter_model()}`")
    else:
        st.caption("Add OPENROUTER_API_KEY to `.env` to enable AI.")

    if st.session_state.last_report:
        pdf_bytes = build_pdf_report(st.session_state.last_report)
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"analysis_report_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

if df is None:
    st.info("Select or upload a dataset to begin.")
    st.stop()

if clean_toggle:
    df, clean_actions = clean_dataframe(df)
else:
    clean_actions = []

profile = profile_dataframe(df)

st.success(f"Loaded **{source_label}** — {profile['rows']:,} rows × {profile['columns']} columns")
if clean_actions:
    with st.expander("Cleaning actions applied"):
        for action in clean_actions:
            st.write(f"- {action}")

tab1, tab2, tab3, tab4 = st.tabs(["Ask a Question", "Chat Mode", "Data Profile", "Raw Preview"])

with tab3:
    st.subheader("Column Overview")
    st.dataframe(pd.DataFrame(profile["column_details"]), use_container_width=True)
    if profile["quality_warnings"]:
        st.warning(" | ".join(profile["quality_warnings"]))
    st.subheader("Column Roles")
    st.json(profile["roles"])

with tab4:
    st.dataframe(df.head(100), use_container_width=True)

with tab1:
    st.subheader("Natural Language Query")
    examples = dataset_examples(profile)

    def _apply_example() -> None:
        if st.session_state.get("example_pick"):
            st.session_state.question_input = st.session_state.example_pick

    st.selectbox(
        "Example questions (auto-fills on select)",
        [""] + examples,
        key="example_pick",
        on_change=_apply_example,
    )

    with st.form("analyze_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            key="question_input",
            height=80,
            placeholder="e.g. Top 10 name by market_cap_usd",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    picked = st.session_state.get("example_pick", "")
    final_question = (question or picked or "").strip()

    if submitted:
        if not final_question:
            st.error("Please type a question or pick an example from the dropdown above.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    report, parser_mode, llm_error = run_query(final_question, df, profile, use_llm)
                st.session_state.last_report = {
                    k: v for k, v in report.items() if k not in {"result_df", "intent_obj"}
                }

                if parser_mode == "OpenRouter AI":
                    st.info(f"Parsed with **OpenRouter AI** ({get_openrouter_model()})")
                elif llm_error:
                    st.warning(f"AI unavailable, using rule-based parser: {llm_error}")

                show_chart(report)
                render_report(report)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    st.markdown("---")
    st.caption(
        "Tip: Pick an example from the dropdown, then click Analyze. "
        "Voice typing: click the box and press **Win + H**."
    )

with tab2:
    st.subheader("Multi-turn Chat")
    st.caption("Ask follow-up questions. Context from prior messages is sent to OpenRouter.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("chart_report"):
                show_chart(msg["chart_report"])

    chat_prompt = st.chat_input("Ask a follow-up question...")
    user_msg = (chat_prompt or "").strip()

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        try:
            with st.spinner("Analyzing..."):
                report, parser_mode, llm_error = run_query(
                    user_msg, df, profile, use_llm, st.session_state.chat_history
                )
            st.session_state.last_report = {
                k: v for k, v in report.items() if k not in {"result_df", "intent_obj"}
            }

            summary = (
                f"**{report['detected_intent']}** (confidence {report['confidence_score']}%)\n\n"
                + "\n".join(f"- {i}" for i in report["analysis"])
            )
            if llm_error and parser_mode != "OpenRouter AI":
                summary = f"_AI fallback used: {llm_error}_\n\n" + summary

            st.session_state.chat_history.append(
                {"role": "assistant", "content": summary, "chart_report": report}
            )
            st.rerun()
        except Exception as exc:
            st.session_state.chat_history.pop()
            st.error(f"Analysis failed: {exc}")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()
