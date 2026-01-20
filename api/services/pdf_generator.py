"""PDF Report Generator Service with Charts."""

import io
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from api.schemas.student import StudentInput, PredictionResponse, RiskFactor


class PDFGeneratorService:
    """Service for generating PDF reports with charts."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#3d2814'),
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#5c3d1e'),
        ))
        self.styles.add(ParagraphStyle(
            name='StudentInfo',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=5,
        ))
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#dc2626'),
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            fontSize=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#f59e0b'),
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#10b981'),
            fontName='Helvetica-Bold',
        ))

    def generate_single_report(
        self,
        prediction: PredictionResponse,
        student: StudentInput
    ) -> bytes:
        """Generate a PDF report for a single student."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
        )

        elements = []

        # Title
        elements.append(Paragraph(
            "Student Dropout Risk Assessment Report",
            self.styles['ReportTitle']
        ))

        # Generation date
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            ParagraphStyle('DateStyle', parent=self.styles['Normal'], alignment=TA_CENTER)
        ))
        elements.append(Spacer(1, 20))

        # Student Information Section
        elements.append(Paragraph("Student Information", self.styles['SectionHeader']))

        info_data = [
            ['Student ID:', student.student_id, 'Program:', student.program],
            ['Enrollment Status:', student.enrollment_status.replace('_', ' ').title(),
             'Enrollment Date:', str(student.enrollment_date)],
            ['GPA:', f'{student.gpa:.2f}', 'Credits Completed:',
             f'{student.credits_completed}/{student.credits_attempted}'],
        ]

        info_table = Table(info_data, colWidths=[1.3*inch, 1.7*inch, 1.3*inch, 1.7*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#5c3d1e')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#5c3d1e')),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))

        # Risk Score Section
        elements.append(Paragraph("Risk Assessment", self.styles['SectionHeader']))

        # Risk status with color
        risk_style = 'RiskHigh' if prediction.risk_status == 'High Risk' else \
                     'RiskMedium' if prediction.risk_status == 'Medium Risk' else 'RiskLow'
        elements.append(Paragraph(
            f"{prediction.risk_percentage}% - {prediction.risk_status}",
            self.styles[risk_style]
        ))
        elements.append(Spacer(1, 10))

        # Charts Section - Gauge and Radar side by side
        elements.append(Paragraph("Risk Visualization", self.styles['SectionHeader']))

        # Create both charts at fixed 300x300px size (3.125 inches)
        chart_size = 3.125 * inch
        gauge_img = self._create_gauge_chart(prediction.risk_score)
        radar_img = self._create_radar_chart(student)

        # Create a table to place charts side by side with 20px (~0.2 inch) gap
        gauge_image = Image(gauge_img, width=chart_size, height=chart_size)
        radar_image = Image(radar_img, width=chart_size, height=chart_size)

        charts_table = Table(
            [[gauge_image, radar_image]],
            colWidths=[chart_size, chart_size],
            rowHeights=[chart_size]
        )
        charts_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (0, 0), 0),
            ('RIGHTPADDING', (0, 0), (0, 0), 10),  # 10pt gap on right of gauge
            ('LEFTPADDING', (1, 0), (1, 0), 10),   # 10pt gap on left of radar
            ('RIGHTPADDING', (1, 0), (1, 0), 0),
        ]))
        elements.append(charts_table)
        elements.append(Spacer(1, 20))

        # Risk Factors Section
        if prediction.risk_factors:
            elements.append(Paragraph("Risk Factors", self.styles['SectionHeader']))
            num_factors = len(prediction.risk_factors)
            risk_factors_img = self._create_risk_factors_chart(prediction.risk_factors)
            chart_height = max(2.5, num_factors * 0.5) * inch
            elements.append(Image(risk_factors_img, width=6.5*inch, height=chart_height))
            elements.append(Spacer(1, 15))

        # Recommendations Section
        if prediction.recommendations:
            elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            for i, rec in enumerate(prediction.recommendations, 1):
                elements.append(Paragraph(
                    f"{i}. {rec}",
                    self.styles['Normal']
                ))
                elements.append(Spacer(1, 5))

        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "Generated by Student Dropout Prediction System",
            ParagraphStyle('Footer', parent=self.styles['Normal'],
                          alignment=TA_CENTER, fontSize=9, textColor=colors.gray)
        ))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    def generate_batch_report(
        self,
        predictions: List[PredictionResponse],
        students: List[StudentInput] = None
    ) -> bytes:
        """Generate a PDF report for batch predictions."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
        )

        elements = []

        # Title
        elements.append(Paragraph(
            "Batch Risk Assessment Summary",
            self.styles['ReportTitle']
        ))

        # Generation date and count
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | "
            f"Total Students: {len(predictions)}",
            ParagraphStyle('DateStyle', parent=self.styles['Normal'], alignment=TA_CENTER)
        ))
        elements.append(Spacer(1, 20))

        # Count risk levels
        high_risk = sum(1 for p in predictions if p.risk_status == 'High Risk')
        medium_risk = sum(1 for p in predictions if p.risk_status == 'Medium Risk')
        low_risk = sum(1 for p in predictions if p.risk_status == 'Low Risk')

        # Summary Statistics
        elements.append(Paragraph("Overview", self.styles['SectionHeader']))

        summary_data = [
            ['High Risk', 'Medium Risk', 'Low Risk', 'Average Score'],
            [
                f'{high_risk} ({high_risk*100//len(predictions)}%)',
                f'{medium_risk} ({medium_risk*100//len(predictions)}%)',
                f'{low_risk} ({low_risk*100//len(predictions)}%)',
                f'{sum(p.risk_percentage for p in predictions)//len(predictions)}%'
            ],
        ]

        summary_table = Table(summary_data, colWidths=[1.5*inch]*4)
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#fee2e2')),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#fef3c7')),
            ('BACKGROUND', (2, 0), (2, 0), colors.HexColor('#d1fae5')),
            ('BACKGROUND', (3, 0), (3, 0), colors.HexColor('#e0e7ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Distribution Pie Chart
        elements.append(Paragraph("Risk Distribution", self.styles['SectionHeader']))
        pie_img = self._create_distribution_pie_chart(high_risk, medium_risk, low_risk)
        elements.append(Image(pie_img, width=5.5*inch, height=4.5*inch))
        elements.append(Spacer(1, 20))

        # Program breakdown if students data available
        if students:
            elements.append(PageBreak())
            elements.append(Paragraph("Risk by Program", self.styles['SectionHeader']))
            program_img = self._create_program_bar_chart(predictions, students)
            elements.append(Image(program_img, width=7*inch, height=4.5*inch))
            elements.append(Spacer(1, 20))

        # High Risk Students Table
        high_risk_students = [p for p in predictions if p.risk_status == 'High Risk']
        if high_risk_students:
            elements.append(Paragraph("High Risk Students", self.styles['SectionHeader']))

            table_data = [['Student ID', 'Risk Score', 'Top Risk Factor']]
            for p in sorted(high_risk_students, key=lambda x: x.risk_score, reverse=True)[:15]:
                top_factor = p.risk_factors[0].factor if p.risk_factors else 'N/A'
                table_data.append([p.student_id, f'{p.risk_percentage}%', top_factor])

            risk_table = Table(table_data, colWidths=[2*inch, 1.2*inch, 3*inch])
            risk_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fecaca')),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(risk_table)

            if len(high_risk_students) > 15:
                elements.append(Paragraph(
                    f"... and {len(high_risk_students) - 15} more high-risk students",
                    ParagraphStyle('More', parent=self.styles['Normal'],
                                  fontSize=9, textColor=colors.gray)
                ))

        # Most Common Risk Factors
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Most Common Risk Factors", self.styles['SectionHeader']))

        all_factors = []
        for p in predictions:
            all_factors.extend([f.factor for f in p.risk_factors])

        factor_counts = Counter(all_factors).most_common(5)
        if factor_counts:
            factor_data = [['Risk Factor', 'Count', 'Percentage']]
            for factor, count in factor_counts:
                factor_data.append([factor, str(count), f'{count*100//len(predictions)}%'])

            factor_table = Table(factor_data, colWidths=[3.5*inch, 1*inch, 1.2*inch])
            factor_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(factor_table)

        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "Generated by Student Dropout Prediction System",
            ParagraphStyle('Footer', parent=self.styles['Normal'],
                          alignment=TA_CENTER, fontSize=9, textColor=colors.gray)
        ))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    def _create_gauge_chart(self, risk_score: float) -> io.BytesIO:
        """Create a semi-circular gauge chart for risk score. Fixed 300x300px output."""
        # Create square figure for 300x300px output at 100 DPI
        fig = plt.figure(figsize=(3, 3), dpi=100)
        ax = fig.add_subplot(111, projection='polar')

        # Configure for semi-circle
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        # Create color zones
        theta_green = np.linspace(np.pi, np.pi * 0.6, 50)
        theta_yellow = np.linspace(np.pi * 0.6, np.pi * 0.3, 50)
        theta_red = np.linspace(np.pi * 0.3, 0, 50)

        # Draw color zones
        ax.fill_between(theta_green, 0.4, 1, color='#10b981', alpha=0.4)
        ax.fill_between(theta_yellow, 0.4, 1, color='#f59e0b', alpha=0.4)
        ax.fill_between(theta_red, 0.4, 1, color='#dc2626', alpha=0.4)

        # Draw needle
        needle_angle = np.pi * (1 - risk_score)
        ax.annotate('', xy=(needle_angle, 0.95), xytext=(needle_angle, 0.15),
                   arrowprops=dict(arrowstyle='->', color='#3d2814', lw=2))

        # Center circle
        circle = plt.Circle((0, 0), 0.12, transform=ax.transData._b,
                           color='#3d2814', zorder=10)
        ax.add_patch(circle)

        # Add percentage text (centered vertically in the gauge)
        ax.text(np.pi/2, -0.15, f'{int(risk_score * 100)}%',
               ha='center', va='center', fontsize=20, fontweight='bold',
               color='#3d2814')

        # Labels at the edges
        ax.text(np.pi, 1.08, '0%', ha='center', va='bottom', fontsize=9)
        ax.text(0, 1.08, '100%', ha='center', va='bottom', fontsize=9)
        ax.text(np.pi/2, 1.08, '50%', ha='center', va='bottom', fontsize=9)

        # Risk status label
        if risk_score >= 0.7:
            status = "High Risk"
            status_color = '#dc2626'
        elif risk_score >= 0.4:
            status = "Medium Risk"
            status_color = '#f59e0b'
        else:
            status = "Low Risk"
            status_color = '#10b981'
        ax.text(np.pi/2, -0.5, status, ha='center', va='center', fontsize=11,
               fontweight='bold', color=status_color)

        # Clean up
        ax.set_ylim(-0.6, 1.15)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)

        # Add title
        ax.set_title('Risk Gauge', fontsize=11, fontweight='bold', pad=5)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer

    def _create_risk_factors_chart(self, factors: List[RiskFactor]) -> io.BytesIO:
        """Create horizontal bar chart for risk factors."""
        fig, ax = plt.subplots(figsize=(12, max(4, len(factors) * 0.8)))

        if not factors:
            ax.text(0.5, 0.5, 'No risk factors identified', ha='center', va='center')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buffer.seek(0)
            return buffer

        # Prepare data
        factor_names = [f.factor for f in factors]
        severities = [f.severity for f in factors]

        # Assign values based on severity
        severity_values = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        values = [severity_values[s] for s in severities]

        # Colors based on severity
        severity_colors = {
            'high': '#dc2626',
            'medium': '#f59e0b',
            'low': '#fbbf24'
        }
        bar_colors = [severity_colors[s] for s in severities]

        # Create horizontal bars
        y_pos = np.arange(len(factor_names))
        bars = ax.barh(y_pos, values, color=bar_colors, height=0.6)

        # Add severity labels
        for i, (bar, severity) in enumerate(zip(bars, severities)):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   severity.upper(), va='center', fontsize=12, fontweight='bold',
                   color=severity_colors[severity])

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factor_names, fontsize=12)
        ax.set_xlim(0, 1.3)
        ax.set_xlabel('Severity', fontsize=12)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)

        plt.tight_layout(pad=2)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.3)
        plt.close()
        buffer.seek(0)
        return buffer

    def _create_radar_chart(self, student: StudentInput) -> io.BytesIO:
        """Create radar chart comparing student metrics to thresholds. Fixed 300x300px output."""
        # Shorter labels to prevent clipping
        categories = ['GPA', 'Attend.', 'LMS', 'Submit.', 'Credits', 'Advisor']

        # Normalize student values (0-1 scale where 1 is good)
        submission_rate = student.assignments_submitted / student.assignments_total if student.assignments_total > 0 else 0
        credit_rate = student.credits_completed / student.credits_attempted if student.credits_attempted > 0 else 0

        student_values = [
            student.gpa / 4.0,  # GPA normalized to 4.0
            student.attendance_rate / 100,  # Already percentage
            min(student.lms_logins_last_30d / 30, 1),  # Normalize to 30 logins
            submission_rate,
            credit_rate,
            min(student.advisor_meetings / 3, 1),  # Normalize to 3 meetings
        ]

        # Threshold values (what's considered "good")
        threshold_values = [
            2.5 / 4.0,  # GPA threshold
            0.80,  # Attendance threshold
            15 / 30,  # LMS logins threshold
            0.80,  # Submission rate threshold
            0.90,  # Credit completion threshold
            1 / 3,  # At least 1 advisor meeting
        ]

        # Number of variables
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Complete the loop
        student_values += student_values[:1]
        threshold_values += threshold_values[:1]
        angles += angles[:1]

        # Create square figure for 300x300px output at 100 DPI
        fig = plt.figure(figsize=(3, 3), dpi=100)
        # Use smaller subplot area (80%) to leave room for labels
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.75], polar=True)

        # Plot threshold area
        ax.fill(angles, threshold_values, color='#10b981', alpha=0.25, label='Threshold')
        ax.plot(angles, threshold_values, color='#10b981', linewidth=1.5, linestyle='--')

        # Plot student values
        ax.fill(angles, student_values, color='#3b82f6', alpha=0.35, label='Student')
        ax.plot(angles, student_values, color='#3b82f6', linewidth=1.5)
        ax.scatter(angles[:-1], student_values[:-1], color='#3b82f6', s=25, zorder=5)

        # Customize - keep labels inside the plot area
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['50%', '100%'], size=6)
        ax.tick_params(pad=2)  # Reduce padding between labels and plot

        # Compact legend inside the figure
        ax.legend(loc='upper right', fontsize=6, framealpha=0.9,
                 bbox_to_anchor=(1.15, 1.0))

        # Add title
        ax.set_title('Student Metrics', fontsize=9, fontweight='bold', pad=8)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        return buffer

    def _create_distribution_pie_chart(
        self, high: int, medium: int, low: int
    ) -> io.BytesIO:
        """Create pie chart for risk distribution."""
        fig, ax = plt.subplots(figsize=(10, 8))

        sizes = [high, medium, low]
        labels = ['High Risk', 'Medium Risk', 'Low Risk']
        colors_list = ['#dc2626', '#f59e0b', '#10b981']
        explode = (0.05, 0.02, 0)

        # Filter out zero values
        non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors_list, explode) if s > 0]
        if not non_zero:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16)
        else:
            sizes, labels, colors_list, explode = zip(*non_zero)

            wedges, texts, autotexts = ax.pie(
                sizes, explode=explode, labels=labels, colors=colors_list,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
                startangle=90, textprops={'fontsize': 14}
            )

            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(13)

            for text in texts:
                text.set_fontsize(14)
                text.set_fontweight('bold')

        ax.axis('equal')
        plt.tight_layout(pad=2)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.3)
        plt.close()
        buffer.seek(0)
        return buffer

    def _create_program_bar_chart(
        self,
        predictions: List[PredictionResponse],
        students: List[StudentInput]
    ) -> io.BytesIO:
        """Create grouped bar chart showing risk by program."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create student_id to program mapping
        program_map = {s.student_id: s.program for s in students}

        # Count risk levels by program
        programs = list(set(program_map.values()))
        program_risks = {p: {'High Risk': 0, 'Medium Risk': 0, 'Low Risk': 0} for p in programs}

        for pred in predictions:
            if pred.student_id in program_map:
                program = program_map[pred.student_id]
                program_risks[program][pred.risk_status] += 1

        # Prepare data for plotting
        x = np.arange(len(programs))
        width = 0.25

        high_counts = [program_risks[p]['High Risk'] for p in programs]
        medium_counts = [program_risks[p]['Medium Risk'] for p in programs]
        low_counts = [program_risks[p]['Low Risk'] for p in programs]

        # Create bars
        bars1 = ax.bar(x - width, high_counts, width, label='High Risk', color='#dc2626')
        bars2 = ax.bar(x, medium_counts, width, label='Medium Risk', color='#f59e0b')
        bars3 = ax.bar(x + width, low_counts, width, label='Low Risk', color='#10b981')

        # Customize
        ax.set_xlabel('Program', fontsize=14)
        ax.set_ylabel('Number of Students', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(programs, rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout(pad=2)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.3)
        plt.close()
        buffer.seek(0)
        return buffer


# Singleton instance
_pdf_generator: PDFGeneratorService | None = None


def get_pdf_generator() -> PDFGeneratorService:
    """Get the singleton PDF generator instance."""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFGeneratorService()
    return _pdf_generator
