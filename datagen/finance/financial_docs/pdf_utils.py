from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from datetime import datetime
import os

def create_pdf_doc(filepath: str) -> SimpleDocTemplate:
    """Create a PDF document with standard settings."""
    return SimpleDocTemplate(
        filepath,
        pagesize=landscape(letter),
        rightMargin=inch/2,
        leftMargin=inch/2,
        topMargin=inch/3,
        bottomMargin=inch/3
    )

def get_title_style() -> ParagraphStyle:
    """Get the standard title style."""
    styles = getSampleStyleSheet()
    return ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER
    )

def get_number_style() -> ParagraphStyle:
    """Get the standard number style."""
    styles = getSampleStyleSheet()
    return ParagraphStyle(
        'Numbers',
        parent=styles['Normal'],
        alignment=TA_RIGHT
    )

def get_footnote_style() -> ParagraphStyle:
    """Get the standard footnote style."""
    styles = getSampleStyleSheet()
    return ParagraphStyle(
        'Footnote',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1  # center alignment
    )

def create_footnote() -> Paragraph:
    """Create a standard footnote."""
    return Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d')} | For illustrative purposes only",
        get_footnote_style()
    )

def style_financial_table(table: Table, table_data: list, sections: dict = None) -> Table:
    """Apply standard styling to a financial table."""
    style = [
        # Headers
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('WORDWRAP', (0, 0), (-1, 0), True),
        # Data
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        # Alternating row colors
        *[('BACKGROUND', (0, i), (-1, i), colors.Color(0.93, 0.93, 0.93)) 
          for i in range(2, len(table_data), 2)],
        # Subtotal lines
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black),
    ]
    
    if sections:
        # Add section separators
        current_col = 1
        for section_fields in sections.values():
            style.append(
                ('LINEAFTER', (current_col + len(section_fields) - 1, 0),
                 (current_col + len(section_fields) - 1, -1), 0.5, colors.black)
            )
            current_col += len(section_fields)
    
    table.setStyle(TableStyle(style))
    return table
