"""End-to-end test script for the data analysis assistant."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analyzer import result_to_dataframe, run_analysis
from src.charts import select_chart
from src.cleaner import clean_dataframe
from src.formatter import format_analysis_report
from src.intent_parser import parse_intent
from src.llm_client import is_llm_configured, parse_question_with_llm
from src.loader import load_file
from src.profiler import profile_dataframe

TESTS = [
    ("usa_states_2014.csv", "Top 10 State by Population"),
    ("ecommerce_products.csv", "Which product category has the highest average price?"),
    ("crypto_top100.csv", "Show correlation between market cap and price"),
    ("weather_daily_2024.csv", "Show trend over time for Mumbai temperature"),
]


def main() -> None:
    print("LLM configured:", is_llm_configured())
    print("=" * 60)
    passed = 0
    for file, question in TESTS:
        df, _ = clean_dataframe(load_file(ROOT / "data" / file))
        profile = profile_dataframe(df)
        intent, err = parse_question_with_llm(question, profile, df)
        mode = "AI" if intent else "RULE"
        if not intent:
            intent = parse_intent(question, profile["roles"], profile["column_names"])
        result, code, sql = run_analysis(df, intent)
        result_df = result_to_dataframe(result)
        chart_type, _ = select_chart(intent, result_df)
        format_analysis_report(question, intent, code, sql, result_df, chart_type, "test", profile)
        ok = len(result_df) > 0
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {file}")
        print(f"  Q: {question}")
        print(f"  Mode: {mode} | Intent: {intent.intent_type} | Rows: {len(result_df)} | Chart: {chart_type}")
        if err and mode == "RULE":
            print(f"  AI fallback: {err[:100]}")
        print(f"  Preview:\n{result_df.head(3).to_string(index=False)}")
        print("-" * 60)
    print(f"Results: {passed}/{len(TESTS)} passed")


if __name__ == "__main__":
    main()
