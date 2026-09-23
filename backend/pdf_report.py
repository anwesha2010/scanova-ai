import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                  Image as RLImage, Table, TableStyle)
import cv2


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "branding", "logo.png")


def _get_logo_flowable():
    """Return logo image, or text fallback if file missing or unreadable."""
    if os.path.exists(LOGO_PATH):
        try:
            return RLImage(LOGO_PATH, width=1.5 * inch, height=0.5 * inch)
        except Exception as e:
            print(f"[PDF] Logo failed to load: {e}")

    # Fallback: blue "S" badge
    badge_style = ParagraphStyle(
        'Badge', parent=getSampleStyleSheet()['BodyText'],
        fontSize=24, leading=28, textColor=colors.white,
        alignment=1, fontName='Helvetica-Bold'
    )
    badge = Paragraph("S", badge_style)
    return Table(
        [[badge]],
        colWidths=[0.7 * inch],
        rowHeights=[0.7 * inch],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#0ea5e9")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ])
    )


def generate_pdf_report(report, overlay_img):
    """Generate a compact, branded PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch
    )

    styles = getSampleStyleSheet()

    # ---------- Styles ----------
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'],
        textColor=colors.HexColor("#0f172a"),
        fontSize=18, spaceAfter=0, spaceBefore=0, leading=22
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['BodyText'],
        textColor=colors.HexColor("#64748b"),
        fontSize=9.5, spaceAfter=0, spaceBefore=2, leading=12
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'],
        textColor=colors.HexColor("#0ea5e9"),
        fontSize=12, spaceAfter=4, spaceBefore=6, leading=15
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['BodyText'],
        fontSize=10, textColor=colors.HexColor("#1e293b"),
        spaceAfter=4, leading=14
    )
    timestamp_style = ParagraphStyle(
        'Timestamp', parent=body_style,
        fontSize=8.5, textColor=colors.HexColor("#94a3b8"),
        spaceAfter=0, leading=11
    )
    footer_style = ParagraphStyle(
        'Footer', parent=body_style,
        fontSize=8, textColor=colors.HexColor("#94a3b8"),
        alignment=1, spaceAfter=0, leading=10
    )

    story = []

    # ==========================================================
    # HEADER — Logo + Title
    # ==========================================================
    logo_flowable = _get_logo_flowable()

    title_block = [
        Paragraph("SCANOVA AI", title_style),
        Paragraph("Medical Image Analysis Report", subtitle_style),
    ]

    header_table = Table(
        [[logo_flowable, title_block]],
        colWidths=[1.7 * inch, 4.6 * inch],
        rowHeights=[0.7 * inch]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)

    # Divider line
    story.append(Spacer(1, 0.08 * inch))
    story.append(Table(
        [['']],
        colWidths=[6.3 * inch],
        rowHeights=[0.015 * inch],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#0ea5e9")),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ])
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Timestamp (tight under divider)
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        timestamp_style
    ))
    story.append(Spacer(1, 0.12 * inch))

    # ==========================================================
    # SUMMARY TABLE
    # ==========================================================
    story.append(Paragraph("Summary", heading_style))

    data = [
        ["Metric", "Value"],
        ["Regions Found", str(report.get("regions_found", 0))],
        ["Affected Area", f"{report.get('affected_area_percent', 0)}%"],
        ["Status", str(report.get("level", "unknown")).upper()],
    ]
    tbl = Table(data, colWidths=[2.2 * inch, 4.1 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.1 * inch))

    # ==========================================================
    # STATUS MESSAGE
    # ==========================================================
    story.append(Paragraph("Status Message", heading_style))
    status_text = report.get("status", "No status available.")
    if not status_text or not str(status_text).strip():
        status_text = "Analysis complete."
    story.append(Paragraph(str(status_text), body_style))
    story.append(Spacer(1, 0.1 * inch))

    # ==========================================================
    # DETECTION OVERLAY IMAGE
    # ==========================================================
    story.append(Paragraph("Detection Overlay", heading_style))

    success, buf = cv2.imencode('.png', overlay_img)
    if success:
        img_bytes = io.BytesIO(buf.tobytes())
        # Fit image to page width with margin
        rl_img = RLImage(img_bytes, width=3.6 * inch, height=3.6 * inch)
        img_table = Table([[rl_img]], colWidths=[6.3 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(img_table)
    else:
        story.append(Paragraph("Image unavailable.", body_style))

    story.append(Spacer(1, 0.12 * inch))

    # ==========================================================
    # DISCLAIMER (with guaranteed fallback text)
    # ==========================================================
    story.append(Paragraph("Disclaimer", heading_style))

    disclaimer_text = report.get("disclaimer", "")
    if not disclaimer_text or not str(disclaimer_text).strip():
        disclaimer_text = (
            "Assistive tool only. Final diagnosis must be made by "
            "a qualified radiologist. SCANOVA AI is not a medical device."
        )
    story.append(Paragraph(str(disclaimer_text), body_style))
        # Extended legal disclaimer
    extended = (
        "<b>Not a medical device.</b> SCANOVA is an educational/research "
        "prototype. It does not replace a radiologist, physician, or other "
        "qualified healthcare professional. Results may be incorrect. "
        "Do not use this report as the sole basis for any medical decision. "
        "Always consult a qualified healthcare professional for diagnosis "
        "and treatment."
    )
    story.append(Paragraph(extended, body_style))

    story.append(Spacer(1, 0.2 * inch))

    # ==========================================================
    # FOOTER
    # ==========================================================
    story.append(Table(
        [['']],
        colWidths=[6.3 * inch],
        rowHeights=[0.01 * inch],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#e4e4e7")),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ])
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "SCANOVA AI · scanova-ai.streamlit.app · MIT Licensed",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()