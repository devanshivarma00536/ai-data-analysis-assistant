"""Export analysis reports to PDF."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "AI Data Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)

    def section(self, title: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)


def _safe(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf_report(report: dict[str, Any]) -> bytes:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.section("User Question")
    pdf.body_text(_safe(str(report.get("user_question", ""))))

    pdf.section("Detected Intent")
    pdf.body_text(_safe(f"{report.get('detected_intent', '')} — {report.get('intent_detail', '')}"))

    pdf.section("Relevant Columns")
    cols = report.get("relevant_columns") or []
    pdf.body_text(_safe(", ".join(cols) if cols else "—"))

    pdf.section("Generated Pandas Code")
    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(0, 4, _safe(str(report.get("pandas_code", ""))))
    pdf.ln(2)

    pdf.section("Generated SQL")
    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(0, 4, _safe(str(report.get("sql_code", ""))))
    pdf.ln(2)

    pdf.section("Visualization")
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(_safe(f"Chart: {report.get('chart_type', '')} — {report.get('chart_reason', '')}"))

    pdf.section("Analysis")
    for item in report.get("analysis", []):
        pdf.body_text(_safe(f"• {item}"))

    pdf.section("Business Insights")
    for item in report.get("business_insights", []):
        pdf.body_text(_safe(f"• {item}"))

    pdf.section("Recommendations")
    for item in report.get("recommendations", []):
        pdf.body_text(_safe(f"• {item}"))

    pdf.section("Confidence Score")
    pdf.body_text(_safe(f"{report.get('confidence_score', 0)}%"))

    pdf.section("Result Preview")
    preview = report.get("result_preview")
    if preview is not None:
        pdf.set_font("Courier", "", 7)
        pdf.multi_cell(0, 3, _safe(preview.head(15).to_string(index=False)))

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, _safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), align="R")

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
