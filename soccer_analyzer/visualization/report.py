"""PDF match report generator."""

import io
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
from datetime import datetime


class MatchReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(230, 237, 243)
        self.cell(0, 10, "U10 Soccer Match Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(139, 148, 158)
        self.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 200, 83)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def generate_pdf_report(match_info: dict, stats_df: pd.DataFrame,
                        figures: dict = None) -> bytes:
    """
    Generate a PDF match report.

    match_info: {date, opponent, result, field_name}
    stats_df: DataFrame with player stats
    figures: dict of {name: matplotlib.Figure} for embedded charts

    Returns: PDF as bytes
    """
    pdf = MatchReportPDF()
    pdf.add_page()

    # Match header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(230, 237, 243)
    opponent = match_info.get("opponent", "Unknown")
    date = match_info.get("date", "")
    result = match_info.get("result", "")
    pdf.cell(0, 12, f"vs {opponent}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(0, 8, f"{date}  |  Result: {result}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Home team stats table
    if not stats_df.empty:
        home_stats = stats_df[stats_df["team"] == "Home"].sort_values("jersey_number")
        away_stats = stats_df[stats_df["team"] == "Away"].sort_values("jersey_number")

        if not home_stats.empty:
            pdf.section_title("Home Team Statistics")
            _add_stats_table(pdf, home_stats)
            pdf.ln(5)

        if not away_stats.empty:
            pdf.section_title("Away Team Statistics")
            _add_stats_table(pdf, away_stats)
            pdf.ln(5)

    # Embedded figures
    if figures:
        temp_dir = tempfile.mkdtemp()
        for name, fig in figures.items():
            if fig is None:
                continue
            try:
                img_path = os.path.join(temp_dir, f"{name}.png")
                fig.savefig(img_path, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                plt.close(fig)

                pdf.add_page()
                pdf.section_title(name.replace("_", " ").title())
                pdf.image(img_path, x=10, w=190)
            except Exception:
                continue

        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Coach notes section
    pdf.add_page()
    pdf.section_title("Coach Notes")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(230, 237, 243)

    notes = match_info.get("notes", "")
    if notes:
        pdf.multi_cell(0, 6, notes)
    else:
        pdf.ln(5)
        for _ in range(8):
            pdf.cell(0, 8, "_" * 95, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def _add_stats_table(pdf, stats_df: pd.DataFrame):
    """Add a stats table to the PDF."""
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(230, 237, 243)

    # Column headers
    columns = ["#", "Name", "Min", "Dist(m)", "Top Spd", "Sprints", "Att%", "Passes"]
    col_widths = [10, 35, 15, 20, 20, 18, 15, 15]

    # Header row
    pdf.set_fill_color(22, 27, 34)
    for i, col in enumerate(columns):
        pdf.cell(col_widths[i], 7, col, border=1, align="C", fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 8)
    for _, row in stats_df.iterrows():
        data = [
            str(int(row.get("jersey_number", 0))),
            str(row.get("name", ""))[:15],
            f"{row.get('minutes_played', 0):.0f}",
            f"{row.get('distance_m', 0):.0f}",
            f"{row.get('top_speed_ms', 0):.1f}",
            str(int(row.get("sprint_count", 0))),
            f"{row.get('pct_att_third', 0):.0f}",
            str(int(row.get("passes_made", 0))),
        ]

        for i, val in enumerate(data):
            pdf.cell(col_widths[i], 6, val, border=1, align="C")
        pdf.ln()
