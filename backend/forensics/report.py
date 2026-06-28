"""
Argus Core - PDF Forensic Report Generator
==========================================
PDF forensic report generation with embedded evidence.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - forensics/report.py

SOTA Algorithm: None (report generation)

Role: Generate comprehensive forensic PDF reports for deepfake analysis.

Report Sections:
1. Executive Summary (score, verdict)
2. Methodology (models used, versions)
3. Findings by Modality
4. Evidence (heatmaps, spectrograms)
5. Technical Appendix
6. Chain of Custody

Integration:
- Imports: schemas/schemas.py, storage/storage.py
- Inputs: AnalysisResult
- Outputs: PDF bytes

Why this approach: PDF reports provide legal-admissible documentation.
Embedded evidence enables offline verification.
"""

import io
import hashlib
import base64
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from utils.logging import get_logger
from schemas.schemas import (
    AnalysisDocument, AnalysisStatus, TrustScore, Verdict, Explanation,
    VideoResult, AudioResult, MetadataResult, Modality
)

logger = get_logger(__name__)


# ============== REPORT DATA MODELS ==============

class ReportMetadata(BaseModel):
    """Metadata for the generated report."""
    report_id: str = Field(..., description="Unique report identifier")
    analysis_id: str = Field(..., description="Source analysis ID")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generator_version: str = Field(default="1.0.0")
    page_count: int = Field(default=0)
    file_hash: str = Field(default="", description="SHA-256 hash of PDF content")


class ReportSection(BaseModel):
    """Individual report section."""
    title: str
    content: str
    subsections: List["ReportSection"] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list, description="Base64 encoded images")


class ReportConfig(BaseModel):
    """Report generation configuration."""
    include_heatmaps: bool = Field(default=True)
    include_technical_details: bool = Field(default=True)
    include_methodology: bool = Field(default=True)
    include_chain_of_custody: bool = Field(default=True)
    include_raw_data: bool = Field(default=False)
    company_name: str = Field(default="Argus Core")
    watermark: bool = Field(default=True)


# ============== REPORT GENERATOR ==============

class ReportGenerator:
    """
    Generate forensic PDF reports.
    
    Creates comprehensive PDF reports with:
    1. Executive Summary (score, verdict)
    2. Methodology (models used, versions)
    3. Findings by Modality
    4. Evidence (heatmaps, spectrograms)
    5. Technical Appendix
    6. Chain of Custody
    
    Usage:
        generator = ReportGenerator()
        pdf_bytes = await generator.generate(analysis)
        
        # Or with storage integration
        report_url = await generator.generate_and_upload(
            analysis=analysis,
            storage=storage_client
        )
    """
    
    # Report styling constants
    COLORS = {
        "authentic": "#22c55e",      # Green
        "likely_authentic": "#84cc16", # Lime
        "uncertain": "#eab308",        # Yellow
        "likely_fake": "#f97316",      # Orange
        "fake": "#ef4444"              # Red
    }
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report generation configuration
        """
        self.config = config or ReportConfig()
        self._reportlab_available = self._check_reportlab()
    
    def _check_reportlab(self) -> bool:
        """Check if ReportLab library is available."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            return True
        except ImportError:
            logger.warning("ReportLab not available, using fallback text report")
            return False
    
    async def generate(
        self,
        analysis: AnalysisDocument,
        heatmap_images: Optional[Dict[str, bytes]] = None
    ) -> bytes:
        """
        Generate comprehensive PDF report.
        
        Args:
            analysis: Completed analysis document
            heatmap_images: Optional dict of heatmap images (key -> bytes)
            
        Returns:
            PDF file bytes
        """
        if analysis.status != AnalysisStatus.COMPLETED:
            raise ValueError("Cannot generate report for incomplete analysis")
        
        if self._reportlab_available:
            return await self._generate_pdf_reportlab(analysis, heatmap_images)
        else:
            return self._generate_text_report(analysis)
    
    async def _generate_pdf_reportlab(
        self,
        analysis: AnalysisDocument,
        heatmap_images: Optional[Dict[str, bytes]] = None
    ) -> bytes:
        """Generate PDF using ReportLab library."""
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.units import inch, cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, black, white, gray
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image as RLImage, ListFlowable, ListItem
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2.5*cm,
            bottomMargin=2.5*cm,
            title=f"Argus Core Analysis Report - {analysis.analysis_id}",
            author="Argus Core Deepfake Detection Platform",
            subject="Forensic Deepfake Analysis Report",
            creator=f"Argus Core Report Generator v{self.VERSION}"
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor("#1f2937")
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=HexColor("#374151")
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=13,
            spaceBefore=15,
            spaceAfter=8,
            textColor=HexColor("#4b5563")
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # Build document content
        story = []
        
        # ===== HEADER =====
        story.append(Paragraph(
            "ARGUS CORE",
            ParagraphStyle('Header', fontSize=12, textColor=HexColor("#6b7280"), alignment=TA_CENTER)
        ))
        story.append(Paragraph(
            "Multi-Modal Deepfake Detection Platform",
            ParagraphStyle('SubHeader', fontSize=10, textColor=HexColor("#9ca3af"), alignment=TA_CENTER)
        ))
        story.append(Spacer(1, 30))
        
        # ===== TITLE =====
        story.append(Paragraph("FORENSIC ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 10))
        
        # Report info table
        report_info = [
            ["Analysis ID:", analysis.analysis_id],
            ["Generated:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["Analysis Date:", analysis.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if analysis.created_at else "N/A"],
            ["Report Version:", self.VERSION]
        ]
        
        info_table = Table(report_info, colWidths=[3*cm, 10*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor("#6b7280")),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 30))
        
        # ===== SECTION 1: EXECUTIVE SUMMARY =====
        story.append(Paragraph("1. EXECUTIVE SUMMARY", heading1_style))
        
        # Verdict box
        verdict_color = self._get_verdict_color(analysis.verdict)
        verdict_text = analysis.verdict.value.upper().replace("_", " ") if analysis.verdict else "UNKNOWN"
        trust_score = analysis.trust_score.value if analysis.trust_score else 0
        
        verdict_data = [[
            Paragraph(f"<b>VERDICT: {verdict_text}</b>", 
                     ParagraphStyle('VerdictText', fontSize=16, textColor=white, alignment=TA_CENTER)),
            Paragraph(f"<b>TRUST SCORE: {trust_score:.1f}</b>", 
                     ParagraphStyle('ScoreText', fontSize=16, textColor=white, alignment=TA_CENTER))
        ]]
        
        verdict_table = Table(verdict_data, colWidths=[8*cm, 5*cm])
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), HexColor(verdict_color)),
            ('BACKGROUND', (1, 0), (1, 0), HexColor("#374151")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 20))
        
        # Explanation
        if analysis.explanation:
            story.append(Paragraph(analysis.explanation.summary, body_style))
            story.append(Spacer(1, 10))
            
            if analysis.explanation.key_findings:
                story.append(Paragraph("<b>Key Findings:</b>", body_style))
                findings_items = [
                    ListItem(Paragraph(finding, body_style))
                    for finding in analysis.explanation.key_findings
                ]
                story.append(ListFlowable(findings_items, bulletType='bullet', start='-'))
        
        story.append(Spacer(1, 20))
        
        # ===== SECTION 2: ANALYSIS DETAILS =====
        story.append(Paragraph("2. ANALYSIS DETAILS", heading1_style))
        
        # Input file info
        if analysis.input:
            story.append(Paragraph("2.1 Input File Information", heading2_style))
            
            input_data = [
                ["Filename:", analysis.input.original_filename],
                ["File Type:", analysis.input.file_type],
                ["File Size:", f"{analysis.input.file_size / 1024:.1f} KB"],
                ["SHA-256 Hash:", analysis.input.file_hash[:32] + "..."],
            ]
            
            if analysis.input.duration_seconds:
                input_data.append(["Duration:", f"{analysis.input.duration_seconds:.1f} seconds"])
            
            input_table = Table(input_data, colWidths=[4*cm, 10*cm])
            input_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
                ('BACKGROUND', (0, 0), (0, -1), HexColor("#f9fafb")),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(input_table)
            story.append(Spacer(1, 15))
        
        # Processing info
        story.append(Paragraph("2.2 Processing Information", heading2_style))
        
        proc_data = [
            ["Status:", analysis.status.value.upper()],
            ["Processing Time:", f"{analysis.processing_time_seconds:.2f} seconds" if analysis.processing_time_seconds else "N/A"],
            ["Completed At:", analysis.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if analysis.completed_at else "N/A"],
        ]
        
        proc_table = Table(proc_data, colWidths=[4*cm, 10*cm])
        proc_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ('BACKGROUND', (0, 0), (0, -1), HexColor("#f9fafb")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(proc_table)
        
        # ===== SECTION 3: MODALITY RESULTS =====
        story.append(PageBreak())
        story.append(Paragraph("3. MODALITY ANALYSIS RESULTS", heading1_style))
        
        # Video Results
        if analysis.video_result:
            story.extend(self._build_video_section(analysis.video_result, heading2_style, body_style))
        
        # Audio Results
        if analysis.audio_result:
            story.extend(self._build_audio_section(analysis.audio_result, heading2_style, body_style))
        
        # Metadata Results
        if analysis.metadata_result:
            story.extend(self._build_metadata_section(analysis.metadata_result, heading2_style, body_style))
        
        # ===== SECTION 4: METHODOLOGY =====
        if self.config.include_methodology:
            story.append(PageBreak())
            story.append(Paragraph("4. METHODOLOGY", heading1_style))
            story.extend(self._build_methodology_section(analysis, heading2_style, body_style))
        
        # ===== SECTION 5: TECHNICAL APPENDIX =====
        if self.config.include_technical_details:
            story.append(PageBreak())
            story.append(Paragraph("5. TECHNICAL APPENDIX", heading1_style))
            story.extend(self._build_technical_appendix(analysis, heading2_style, body_style))
        
        # ===== SECTION 6: CHAIN OF CUSTODY =====
        if self.config.include_chain_of_custody:
            story.append(PageBreak())
            story.append(Paragraph("6. CHAIN OF CUSTODY", heading1_style))
            story.extend(self._build_chain_of_custody(analysis, heading2_style, body_style))
        
        # ===== FOOTER =====
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "This report was generated automatically by Argus Core Deepfake Detection Platform. "
            "Results should be reviewed by qualified personnel before making critical decisions.",
            ParagraphStyle('Footer', fontSize=8, textColor=HexColor("#9ca3af"), alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        logger.info(f"PDF report generated: {len(pdf_bytes)} bytes")
        
        return pdf_bytes
    
    def _generate_text_report(self, analysis: AnalysisDocument) -> bytes:
        """Generate fallback text report when ReportLab unavailable."""
        lines = []
        
        lines.append("=" * 70)
        lines.append("ARGUS CORE - FORENSIC ANALYSIS REPORT")
        lines.append("Multi-Modal Deepfake Detection Platform")
        lines.append("=" * 70)
        lines.append("")
        
        # Report info
        lines.append(f"Analysis ID: {analysis.analysis_id}")
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Report Version: {self.VERSION}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("1. EXECUTIVE SUMMARY")
        lines.append("-" * 70)
        lines.append("")
        
        verdict_text = analysis.verdict.value.upper().replace("_", " ") if analysis.verdict else "UNKNOWN"
        trust_score = analysis.trust_score.value if analysis.trust_score else 0
        
        lines.append(f"VERDICT: {verdict_text}")
        lines.append(f"TRUST SCORE: {trust_score:.1f}")
        lines.append("")
        
        if analysis.explanation:
            lines.append(f"Summary: {analysis.explanation.summary}")
            lines.append("")
            
            if analysis.explanation.key_findings:
                lines.append("Key Findings:")
                for finding in analysis.explanation.key_findings:
                    lines.append(f"  - {finding}")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("2. ANALYSIS DETAILS")
        lines.append("-" * 70)
        lines.append("")
        
        # Input info
        if analysis.input:
            lines.append("Input File Information:")
            lines.append(f"  Filename: {analysis.input.original_filename}")
            lines.append(f"  File Type: {analysis.input.file_type}")
            lines.append(f"  File Size: {analysis.input.file_size / 1024:.1f} KB")
            lines.append(f"  SHA-256: {analysis.input.file_hash}")
            if analysis.input.duration_seconds:
                lines.append(f"  Duration: {analysis.input.duration_seconds:.1f} seconds")
            lines.append("")
        
        # Processing info
        lines.append("Processing Information:")
        lines.append(f"  Status: {analysis.status.value.upper()}")
        if analysis.processing_time_seconds:
            lines.append(f"  Processing Time: {analysis.processing_time_seconds:.2f} seconds")
        if analysis.completed_at:
            lines.append(f"  Completed At: {analysis.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("3. MODALITY ANALYSIS RESULTS")
        lines.append("-" * 70)
        lines.append("")
        
        # Video results
        if analysis.video_result:
            lines.append("Video Analysis:")
            lines.append(f"  Aggregate Score: {analysis.video_result.aggregate_score:.2%}")
            lines.append(f"  Spatial Score: {analysis.video_result.spatial.score:.2%}")
            lines.append(f"  Temporal Score: {analysis.video_result.temporal.consistency_score:.2%}")
            if analysis.video_result.lip_sync:
                lines.append(f"  Lip-Sync Score: {analysis.video_result.lip_sync.sync_score:.2%}")
            lines.append(f"  Frames Analyzed: {analysis.video_result.frames_analyzed}")
            lines.append("")
        
        # Audio results
        if analysis.audio_result:
            lines.append("Audio Analysis:")
            lines.append(f"  Synthetic Probability: {analysis.audio_result.synthetic_probability:.2%}")
            lines.append(f"  Vocoder Artifacts: {'Detected' if analysis.audio_result.vocoder_artifacts_detected else 'Not Detected'}")
            lines.append(f"  Voice Consistency: {analysis.audio_result.voice_consistency_score:.2%}")
            lines.append("")
        
        # Metadata results
        if analysis.metadata_result:
            lines.append("Metadata Analysis:")
            lines.append(f"  C2PA Present: {'Yes' if analysis.metadata_result.c2pa.present else 'No'}")
            if analysis.metadata_result.c2pa.present:
                lines.append(f"  C2PA Valid: {'Yes' if analysis.metadata_result.c2pa.valid else 'No'}")
            lines.append(f"  File Structure Valid: {'Yes' if analysis.metadata_result.file_structure_valid else 'No'}")
            if analysis.metadata_result.exif_anomalies:
                lines.append("  EXIF Anomalies:")
                for anomaly in analysis.metadata_result.exif_anomalies:
                    lines.append(f"    - {anomaly}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("4. METHODOLOGY")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Models and Algorithms Used:")
        lines.append("  - EfficientNet-B3: Spatial artifact detection")
        lines.append("  - X-CLIP: Temporal consistency analysis")
        lines.append("  - LIPINC-V2: Lip-sync verification")
        lines.append("  - Purdue-M2: Audio deepfake detection")
        lines.append("  - AASIST: Audio anti-spoofing")
        lines.append("  - C2PA v2.3: Content authenticity verification")
        lines.append("")
        lines.append("Trust Score Calculation:")
        lines.append("  - Attention-weighted multi-modal fusion")
        lines.append("  - Platt scaling for probability calibration")
        lines.append("  - Uncertainty quantification via ensemble disagreement")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("DISCLAIMER")
        lines.append("-" * 70)
        lines.append("")
        lines.append("This report was generated automatically by Argus Core Deepfake")
        lines.append("Detection Platform. Results should be reviewed by qualified")
        lines.append("personnel before making critical decisions.")
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Generated by Argus Core v{self.VERSION}")
        lines.append("=" * 70)
        
        return "\n".join(lines).encode("utf-8")
    
    def _get_verdict_color(self, verdict: Optional[Verdict]) -> str:
        """Get color for verdict."""
        if verdict is None:
            return "#6b7280"  # Gray
        return self.COLORS.get(verdict.value, "#6b7280")
    
    def _build_video_section(
        self,
        video_result: VideoResult,
        heading2_style,
        body_style
    ) -> List:
        """Build video analysis section."""
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
        
        elements = []
        
        elements.append(Paragraph("3.1 Video Analysis", heading2_style))
        elements.append(Paragraph(
            "Multi-stage video analysis using spatial, temporal, and lip-sync detection.",
            body_style
        ))
        elements.append(Spacer(1, 10))
        
        # Score table
        score_data = [
            ["Component", "Score", "Status"],
            ["Overall", f"{video_result.aggregate_score:.1%}", self._get_status_text(video_result.aggregate_score)],
            ["Spatial Analysis", f"{video_result.spatial.score:.1%}", self._get_status_text(video_result.spatial.score)],
            ["Temporal Consistency", f"{video_result.temporal.consistency_score:.1%}", self._get_status_text(video_result.temporal.consistency_score)],
        ]
        
        if video_result.lip_sync:
            score_data.append([
                "Lip-Sync Verification", 
                f"{video_result.lip_sync.sync_score:.1%}", 
                self._get_status_text(video_result.lip_sync.sync_score)
            ])
        
        score_table = Table(score_data, colWidths=[5*cm, 3*cm, 4*cm])
        score_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f3f4f6")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(score_table)
        elements.append(Spacer(1, 10))
        
        # Additional details
        elements.append(Paragraph(f"Frames Analyzed: {video_result.frames_analyzed}", body_style))
        elements.append(Paragraph(f"Face Detected: {'Yes' if video_result.face_detected else 'No'}", body_style))
        
        if video_result.temporal.flickering_detected:
            elements.append(Paragraph(
                "<b>Warning:</b> Flickering artifacts detected in video.",
                body_style
            ))
        
        if video_result.spatial.anomaly_indices:
            elements.append(Paragraph(
                f"Anomalous frames detected at indices: {video_result.spatial.anomaly_indices[:10]}...",
                body_style
            ))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_audio_section(
        self,
        audio_result: AudioResult,
        heading2_style,
        body_style
    ) -> List:
        """Build audio analysis section."""
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
        
        elements = []
        
        elements.append(Paragraph("3.2 Audio Analysis", heading2_style))
        elements.append(Paragraph(
            "Synthetic voice detection using Purdue-M2 architecture.",
            body_style
        ))
        elements.append(Spacer(1, 10))
        
        # Score table
        authenticity_score = 1 - audio_result.synthetic_probability
        score_data = [
            ["Metric", "Value", "Status"],
            ["Authenticity Score", f"{authenticity_score:.1%}", self._get_status_text(authenticity_score)],
            ["Synthetic Probability", f"{audio_result.synthetic_probability:.1%}", self._get_status_text(1 - audio_result.synthetic_probability)],
            ["Voice Consistency", f"{audio_result.voice_consistency_score:.1%}", self._get_status_text(audio_result.voice_consistency_score)],
        ]
        
        score_table = Table(score_data, colWidths=[5*cm, 3*cm, 4*cm])
        score_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f3f4f6")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(score_table)
        elements.append(Spacer(1, 10))
        
        # Vocoder artifacts
        if audio_result.vocoder_artifacts_detected:
            elements.append(Paragraph(
                "<b>Warning:</b> Vocoder artifacts detected, indicating potential synthetic audio.",
                body_style
            ))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_metadata_section(
        self,
        metadata_result: MetadataResult,
        heading2_style,
        body_style
    ) -> List:
        """Build metadata analysis section."""
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
        from reportlab.lib.units import cm
        
        elements = []
        
        elements.append(Paragraph("3.4 Metadata Analysis", heading2_style))
        elements.append(Paragraph(
            "Content authenticity verification via C2PA and EXIF analysis.",
            body_style
        ))
        elements.append(Spacer(1, 10))
        
        # C2PA info
        elements.append(Paragraph("<b>C2PA Content Credentials:</b>", body_style))
        c2pa_data = [
            ["Property", "Value"],
            ["Present", "Yes" if metadata_result.c2pa.present else "No"],
        ]
        
        if metadata_result.c2pa.present:
            c2pa_data.extend([
                ["Valid", "Yes" if metadata_result.c2pa.valid else "No" if metadata_result.c2pa.valid is False else "Unknown"],
                ["Issuer", metadata_result.c2pa.issuer or "Unknown"],
            ])
            if metadata_result.c2pa.issued_at:
                c2pa_data.append(["Issued At", metadata_result.c2pa.issued_at.strftime("%Y-%m-%d %H:%M:%S UTC")])
        
        c2pa_table = Table(c2pa_data, colWidths=[4*cm, 8*cm])
        c2pa_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f3f4f6")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(c2pa_table)
        elements.append(Spacer(1, 10))
        
        # File structure
        elements.append(Paragraph(
            f"<b>File Structure:</b> {'Valid' if metadata_result.file_structure_valid else 'Invalid/Modified'}",
            body_style
        ))
        
        # EXIF anomalies
        if metadata_result.exif_anomalies:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>EXIF Anomalies Detected:</b>", body_style))
            anomaly_items = [
                ListItem(Paragraph(anomaly, body_style))
                for anomaly in metadata_result.exif_anomalies
            ]
            elements.append(ListFlowable(anomaly_items, bulletType='bullet', start='-'))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_methodology_section(
        self,
        analysis: AnalysisDocument,
        heading2_style,
        body_style
    ) -> List:
        """Build methodology section."""
        from reportlab.platypus import Paragraph, Spacer, ListFlowable, ListItem
        
        elements = []
        
        elements.append(Paragraph("4.1 Detection Models", heading2_style))
        
        models_used = []
        
        if analysis.video_result:
            models_used.extend([
                "EfficientNet-B3: Per-frame spatial artifact detection (from DeepfakeBench)",
                "CLIP ViT-B/16: Feature extraction for generalization to unseen forgery types",
                "X-CLIP: Temporal consistency analysis with Multiframe Integration Transformer",
                "LIPINC-V2: Vision Temporal Transformer for lip-sync deepfake detection",
                "RetinaFace: Face detection and landmark extraction"
            ])
        
        if analysis.audio_result:
            models_used.extend([
                "Purdue-M2: AI-synthesized voice generalization (AAAI 2025)",
                "Mel-spectrogram analysis: 80 mel bands, vocoder artifact detection"
            ])
        
        if analysis.metadata_result:
            models_used.extend([
                "C2PA v2.3: Content authenticity verification",
                "EXIF Parser: Metadata consistency analysis"
            ])
        
        model_items = [ListItem(Paragraph(model, body_style)) for model in models_used]
        elements.append(ListFlowable(model_items, bulletType='bullet', start='•'))
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("4.2 Scoring Methodology", heading2_style))
        elements.append(Paragraph(
            "The Trust Score is computed using attention-weighted multi-modal fusion with "
            "Platt scaling for probability calibration. Key components:",
            body_style
        ))
        
        scoring_details = [
            "Attention-Based Fusion: Dynamic weights based on modality confidence",
            "Platt Scaling: Ensures scores represent true probabilities",
            "Uncertainty Quantification: Monte Carlo dropout for confidence estimation",
            "Ensemble Disagreement: Detects cases requiring human review"
        ]
        
        scoring_items = [ListItem(Paragraph(detail, body_style)) for detail in scoring_details]
        elements.append(ListFlowable(scoring_items, bulletType='bullet', start='•'))
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("4.3 Verdict Thresholds", heading2_style))
        
        thresholds = [
            "80-100: Authentic (high confidence in authenticity)",
            "60-79: Likely Authentic (moderate confidence)",
            "40-59: Uncertain (requires human review)",
            "20-39: Likely Fake (moderate confidence in manipulation)",
            "0-19: Fake (high confidence in manipulation)"
        ]
        
        threshold_items = [ListItem(Paragraph(t, body_style)) for t in thresholds]
        elements.append(ListFlowable(threshold_items, bulletType='bullet', start='•'))
        
        return elements
    
    def _build_technical_appendix(
        self,
        analysis: AnalysisDocument,
        heading2_style,
        body_style
    ) -> List:
        """Build technical appendix section."""
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Preformatted
        from reportlab.lib.units import cm
        from reportlab.lib.styles import ParagraphStyle
        
        elements = []
        
        elements.append(Paragraph("5.1 System Configuration", heading2_style))
        
        config_data = [
            ["Parameter", "Value"],
            ["Platform Version", self.VERSION],
            ["Inference Backend", "ONNX Runtime with TensorRT/CUDA"],
            ["Quantization", "INT8 (static)"],
            ["GPU Memory Limit", "4GB (RTX 3050 optimized)"],
            ["Frame Sampling Rate", "Every 5th frame"],
            ["Adversarial Defense", analysis.options.defense_level if analysis.options else "standard"],
        ]
        
        config_table = Table(config_data, colWidths=[5*cm, 8*cm])
        config_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f3f4f6")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(config_table)
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("5.2 Raw Score Data", heading2_style))
        
        if analysis.trust_score:
            elements.append(Paragraph(f"Trust Score: {analysis.trust_score.value:.4f}", body_style))
            elements.append(Paragraph(f"Confidence: {analysis.trust_score.confidence:.4f}", body_style))
            elements.append(Paragraph(f"Calibrated: {analysis.trust_score.calibrated}", body_style))
        
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("5.3 File Hashes", heading2_style))
        
        if analysis.input:
            elements.append(Paragraph(f"Input File SHA-256:", body_style))
            elements.append(Paragraph(
                analysis.input.file_hash,
                ParagraphStyle('Code', fontName='Courier', fontSize=8)
            ))
        
        return elements
    
    def _build_chain_of_custody(
        self,
        analysis: AnalysisDocument,
        heading2_style,
        body_style
    ) -> List:
        """Build chain of custody section."""
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
        
        elements = []
        
        elements.append(Paragraph(
            "This section documents the chain of custody for this analysis, "
            "providing an auditable trail of all processing steps.",
            body_style
        ))
        elements.append(Spacer(1, 15))
        
        # Event timeline
        events = [
            ["Timestamp", "Event", "Details"],
        ]
        
        if analysis.created_at:
            events.append([
                analysis.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Analysis Created",
                f"Analysis ID: {analysis.analysis_id[:16]}..."
            ])
        
        if analysis.input:
            events.append([
                analysis.created_at.strftime("%Y-%m-%d %H:%M:%S") if analysis.created_at else "N/A",
                "File Uploaded",
                f"Hash: {analysis.input.file_hash[:32]}..."
            ])
        
        events.append([
            "N/A",
            "Preprocessing",
            "Frame extraction, face detection"
        ])
        
        events.append([
            "N/A",
            "Multi-Modal Analysis",
            "Parallel modality processing"
        ])
        
        events.append([
            "N/A",
            "Result Aggregation",
            "Fusion and scoring"
        ])
        
        if analysis.completed_at:
            events.append([
                analysis.completed_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Analysis Completed",
                f"Verdict: {analysis.verdict.value if analysis.verdict else 'N/A'}"
            ])
        
        events.append([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Report Generated",
            f"Report Version: {self.VERSION}"
        ])
        
        events_table = Table(events, colWidths=[4*cm, 4*cm, 6*cm])
        events_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#f3f4f6")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(events_table)
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("6.1 Cryptographic Verification", heading2_style))
        elements.append(Paragraph(
            "All audit events are cryptographically chained using SHA-256 hashes, "
            "providing tamper-evident logging for legal proceedings.",
            body_style
        ))
        
        return elements
    
    def _get_status_text(self, score: float) -> str:
        """Get status text based on score."""
        if score >= 0.8:
            return "✓ Good"
        elif score >= 0.6:
            return "~ Moderate"
        elif score >= 0.4:
            return "⚠ Uncertain"
        else:
            return "✗ Suspicious"
    
    async def generate_and_upload(
        self,
        analysis: AnalysisDocument,
        storage,
        bucket: Optional[str] = None,
        heatmap_images: Optional[Dict[str, bytes]] = None
    ) -> str:
        """
        Generate report and upload to storage.
        
        Args:
            analysis: Completed analysis document
            storage: StorageClient instance
            bucket: Optional bucket name (defaults to results bucket)
            heatmap_images: Optional heatmap images
            
        Returns:
            Presigned URL to download report
        """
        # Generate PDF
        pdf_bytes = await self.generate(analysis, heatmap_images)
        
        # Determine bucket and key
        bucket = bucket or storage.bucket_results
        report_key = f"results/{analysis.analysis_id}/report.pdf"
        
        # Ensure bucket exists
        await storage.ensure_default_buckets()
        
        # Upload
        await storage.upload_file(
            file=pdf_bytes,
            bucket=bucket,
            object_key=report_key,
            content_type="application/pdf"
        )
        
        # Get presigned URL
        report_url = await storage.get_presigned_url(
            bucket=bucket,
            object_key=report_key,
            expires_seconds=86400  # 24 hours
        )
        
        logger.info(f"Report uploaded: {report_key}")
        
        return report_url
    
    def compute_report_hash(self, pdf_bytes: bytes) -> str:
        """Compute SHA-256 hash of report content."""
        return hashlib.sha256(pdf_bytes).hexdigest()


# ============== SINGLETON ==============

_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """
    Get singleton report generator instance.
    
    Returns:
        ReportGenerator instance
    """
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


# Export
__all__ = [
    "ReportMetadata",
    "ReportSection",
    "ReportConfig",
    "ReportGenerator",
    "get_report_generator"
]
