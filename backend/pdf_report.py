import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                  Image as RLImage, Table, TableStyle)
import cv2


def generate_pdf_report(report, overlay_img):
    """Generate a PDF report with summary and image. Returns bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'],
        textColor=colors.HexColor("#0ea5e9"),
        fontSize=22, spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'],
        textColor=colors.HexColor("#0f172a"),
        fontSize=13, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['BodyText'],
        fontSize=11, textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    story = []

    # Title
    story.append(Paragraph("SCANOVA AI — Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body_style
    ))
    story.append(Spacer(1, 0.3 * inch))

    # Summary table
    story.append(Paragraph("Summary", heading_style))

    data = [
        ["Metric", "Value"],
        ["Regions Found", str(report["regions_found"])],
        ["Affected Area", f"{report['affected_area_percent']}%"],
        ["Status", report["level"].upper()],
    ]
    tbl = Table(data, colWidths=[2 * inch, 3 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.25 * inch))

    # Status message
    story.append(Paragraph("Status Message", heading_style))
    story.append(Paragraph(report["status"], body_style))
    story.append(Spacer(1, 0.25 * inch))

    # Detection Overlay image
    story.append(Paragraph("Detection Overlay", heading_style))

    # Convert numpy BGR image to PNG bytes
        # Convert BGR to RGB for correct colors in PDF
    rgb_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    success, buf = cv2.imencode('.png', rgb_img)
    if success:
        img_bytes = io.BytesIO(buf.tobytes())
        rl_img = RLImage(img_bytes, width=4 * inch, height=4 * inch)
        story.append(rl_img)

    story.append(Spacer(1, 0.25 * inch))

    # Disclaimer
    story.append(Paragraph("Disclaimer", heading_style))
    story.append(Paragraph(report["disclaimer"], body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()