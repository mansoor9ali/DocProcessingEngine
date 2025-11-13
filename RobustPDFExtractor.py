#!/usr/bin/env python

"""
Production-Grade PDF Extractor for RAG Systems
Implements a multi-stage fallback strategy with a specialist router
for technical documents.

Dependencies:
- pypdf
- pdfplumber
- boto3
- amazon-textract-response-parser
- nltk
- gibberish-detector
- marker-pdf (Optional, for equation extraction)
- (and marker's dependencies, e.g., torch, transformers)
"""

import sys
import re
import boto3
import pdfplumber
import nltk
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from botocore.exceptions import ClientError
from gibberish_detector import detector
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from typing import List, Optional



# --- 1. Standardized Data Structures ---

@dataclass
class ExtractedTable:
    """A standardized representation of an extracted table."""
    page_number: int
    as_list: List[List[Optional[str]]]
    as_markdown: str
    confidence: float = 1.0
    extractor: str = ""


@dataclass
class ExtractedHeader:
    """A standardized representation of an inferred document header."""
    text: str
    page_number: int
    level: int  # e.g., 1 for H1, 2 for H2
    font_size: float
    font_name: str
    extractor: str = ""


@dataclass
class ExtractedEquation:
    """A standardized representation of an extracted equation."""
    page_number: int
    as_latex: str
    as_image: Optional[bytes] = None
    extractor: str = ""


@dataclass
class ExtractionResult:
    """
    A unified data structure to hold the output of any extraction stage.
    """
    full_text: str
    page_texts: Dict[int, str]
    tables: List = field(default_factory=list)
    headers: List[ExtractedHeader] = field(default_factory=list)
    equations: List[ExtractedEquation] = field(default_factory=list)

    extractor_used: str
    confidence_score: float
    total_pages: int
    raw_response: Optional[Any] = None




class RobustPDFExtractor:
    """
    Implements the multi-stage fallback pattern for PDF extraction.

    1. (Router): Check if technical doc -> Route to Marker
    2. Stage 1: Try fast `pypdf` extraction.
    3. Stage 2: If low confidence, fall back to `pdfplumber`.
    4. Stage 3: If low confidence, fall back to `AWS Textract`.
    """

    def __init__(self, aws_region_name: str = 'us-east-1', use_marker: bool = False):
        """
        Initializes the extractor, setting up clients and heuristics.
        """
        self.textract_client = boto3.client('textract', region_name=aws_region_name)
        self._initialize_stage1_heuristics()

        self.use_marker = use_marker
        self.marker_models = None
        if self.use_marker:
            try:
                from marker.models import load_all_models
                print("Loading Marker models (this may take a while)...")
                self.marker_models = load_all_models()
                print("Marker models loaded.")
            except ImportError:
                print("Warning: 'marker-pdf' not installed. Marker extraction will be disabled.")
                print("Install with: pip install marker-pdf")
                self.use_marker = False
            except Exception as e:
                print(f"Error loading Marker models: {e}")
                self.use_marker = False

    def _initialize_stage1_heuristics(self):
        """Pre-loads resources needed for confidence scoring."""
        try:
            nltk.data.find('corpus/words')
        except LookupError:
            print("NLTK 'words' corpus not found. Downloading...")
            nltk.download('words')

        self.english_words = set(nltk.corpus.words.words())
        self.gibberish_detector = detector.Detector.create()

    # --- 5. Specialist Router and Marker Implementation ---

    def _is_technical_doc(self, pdf_path: str, reader: PdfReader) -> bool:
        """Heuristic to detect if a PDF is a technical/academic paper."""
        if "arxiv" in pdf_path.lower(): [26]
        return True

    try:
        first_page_text = reader.pages.extract_text()
        if first_page_text:
            text_lower = first_page_text.lower()
            if "abstract" in text_lower and "introduction" in text_lower:
                return True
            if "references" in text_lower and "doi:" in text_lower:
                return True
    except Exception:
        pass
    return False


def _parse_marker_markdown(self, markdown: str) -> ExtractionResult:
    """Parses the Markdown output from Marker."""
    page_texts = {}
    tables =
    headers =
    equations =

    header_re = re.compile(r"^(#{1,6})\s(.*)", re.MULTILINE)
    equation_re = re.compile(r"^\$\$\n(.*?)\n\$\$$", re.DOTALL | re.MULTILINE)

    for match in header_re.finditer(markdown):
        headers.append(ExtractedHeader(
            text=match.group(2).strip(), page_number=0, level=len(match.group(1)),
            font_size=16 - (len(match.group(1)) * 2), font_name="Marker-MD", extractor="marker"
        ))

    for match in equation_re.finditer(markdown):
        equations.append(ExtractedEquation(
            page_number=0, as_latex=match.group(1).strip(), extractor="marker"
        ))

    table_re = re.compile(r"\|.*?\n\|[\s\-:\|]+.*?\n(\|.*?\n)+", re.MULTILINE)
    for i, match in enumerate(table_re.finditer(markdown)):
        md_table = match.group(0).strip()
        tbl_list = [cell.strip() for cell in row.split('|')[1:-1]]
        for row in md_table.split('\n')
    ]
    if len(tbl_list) > 1:
        tbl_list.pop(1)  # Remove separator row
    tables.append(ExtractedTable(
        page_number=0, as_list=tbl_list, as_markdown=md_table, extractor="marker"
    ))


return ExtractionResult(
    full_text=markdown, page_texts={1: markdown}, tables=tables,
    headers=headers, equations=equations, extractor_used="marker",
    confidence_score=0.95, total_pages=1  # Page metadata is lost
)


def _marker_extract(self, pdf_path: str) -> ExtractionResult:
    """Stage 4 (Specialist): Use Marker for academic docs."""
    if not self.use_marker or self.marker_models is None:
        return ExtractionResult(
            full_text="", page_texts={}, extractor_used="marker_skipped",
            confidence_score=0.0, total_pages=0
        )
    try:
        from marker.convert import convert_pdf
        markdown_text, out_meta = convert_pdf(pdf_path, self.marker_models)
        return self._parse_marker_markdown(markdown_text)
    except Exception as e:
        print(f"Marker extraction failed: {e}")
        return ExtractionResult(
            full_text="", page_texts={}, extractor_used="marker_failed",
            confidence_score=0.0, total_pages=0, raw_response=str(e)
        )


# --- 2. Stage 1: pypdf Implementation ---

def _pypdf_extract(self, pdf_path: str) -> (ExtractionResult, Optional):
    """Stage 1: Fast text extraction using pypdf."""
    all_text =
    page_texts = {}
    total_pages = 0
    reader = None

    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            try:
                text = page.extract_text()
                if text is None: text = ""
            except Exception:
                text = ""
            all_text.append(text)
            page_texts[page_num] = text

        full_text = "\n\n".join(all_text)
        confidence = self._calculate_pypdf_confidence(reader, full_text, page_texts)

        return ExtractionResult(
            full_text=full_text, page_texts=page_texts, extractor_used="pypdf",
            confidence_score=confidence, total_pages=total_pages
        ), reader

    except (PdfReadError, Exception) as e:
        print(f"pypdf failed to read file: {e}")
        return ExtractionResult(
            full_text="", page_texts={}, extractor_used="pypdf",
            confidence_score=0.0, total_pages=0
        ), None


# --- 2.2 Stage 1: Confidence Heuristics ---

def _check_for_scanned_pages(self, reader: PdfReader, page_texts: Dict[int, str]) -> float:
    """Detects scanned pages."""
    scanned_page_count = 0
    total_pages = len(reader.pages)
    if total_pages == 0: return 0.0

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        text_length = len(page_texts.get(page_num, "").strip())
        has_images = len(page.images) > 0
        if text_length < 50 and has_images:
            scanned_page_count += 1

    scanned_ratio = scanned_page_count / total_pages
    return 1.0 - scanned_ratio


def _check_for_garbled_text(self, full_text: str) -> float:
    """Detects garbled text."""
    text_to_check = re.sub(r'\s+', ' ', full_text).strip()
    if len(text_to_check) < 100: return 1.0
    if self.gibberish_detector.is_gibberish(text_to_check):
        return 0.0

    words = set(text_to_check.lower().split())
    if not words: return 1.0
    intersection = words.intersection(self.english_words)
    word_ratio = len(intersection) / len(words)
    return 0.3 if word_ratio < 0.3 else 1.0


def _check_for_layout_collapse(self, page_texts: Dict[int, str]) -> float:
    """Detects multi-column layout collapse."""
    total_chars = 0
    total_newlines = 0
    for text in page_texts.values():
        total_chars += len(text)
        total_newlines += text.count('\n')

    if total_chars == 0 or total_newlines == 0: return 1.0
    chars_per_newline = total_chars / total_newlines
    return 0.5 if chars_per_newline > 300 else 1.0


def _calculate_pypdf_confidence(self, reader: PdfReader, full_text: str, page_texts: Dict[int, str]) -> float:
    """Combines all Stage 1 heuristics."""
    scanned_score = self._check_for_scanned_pages(reader, page_texts)
    garbled_score = self._check_for_garbled_text(full_text)
    layout_score = self._check_for_layout_collapse(page_texts)

    if garbled_score == 0.0: return 0.0
    return (scanned_score * 0.6) + (garbled_score * 0.3) + (layout_score * 0.1)


# --- 3. Stage 2: pdfplumber Implementation ---

def _convert_table_to_markdown(self, table_data: List[List[Optional[str]]]) -> str:
    """Converts a list-of-lists table into a Markdown string."""
    if not table_data: return ""
    header = "| " + " | ".join(str(h) if h is not None else "" for h in table_data) + " |"
    separator = "| " + " | ".join(["---"] * len(table_data)) + " |"
    rows = [
        "| " + " | ".join(str(cell) if cell is not None else "" for cell in row) + " |"
        for row in table_data[1:]
    return "\n".join([header, separator] + rows)


def _pdfplumber_extract(self, pdf_path: str) -> ExtractionResult:
    """Stage 2: Structure-aware extraction using pdfplumber."""
    all_text =
    page_texts = {}
    tables =
    headers =
    total_pages = 0
    table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 4}[32]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text is None: text = ""
                all_text.append(text)
                page_texts[page_num] = text

                extracted_tables = page.extract_tables(table_settings=table_settings)[21]
                for tbl_list in extracted_tables:
                    if not tbl_list: continue
                    md_table = self._convert_table_to_markdown(tbl_list)
                    tables.append(ExtractedTable(
                        page_number=page_num, as_list=tbl_list, as_markdown=md_table, extractor="pdfplumber"
                    ))
                headers.extend(self._find_headers_with_pdfplumber(page))

        full_text = "\n\n".join(all_text)
        confidence = self._calculate_pdfplumber_confidence(full_text)

        return ExtractionResult(
            full_text=full_text, page_texts=page_texts, tables=tables,
            headers=headers, extractor_used="pdfplumber",
            confidence_score=confidence, total_pages=total_pages
        )
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        return ExtractionResult(
            full_text="", page_texts={}, extractor_used="pdfplumber",
            confidence_score=0.0, total_pages=0
        )


# --- 3.2 Stage 2: Header Detection Logic ---

def _get_body_font_stats(self, page: pdfplumber.page.Page) -> (float, str):
    """Helper to find the most common (body) font size and name.[37]"""
    try:
        sizes = [char['size'] for char in page.chars if 'size' in char and char['text'].strip()]
        names = [char['fontname'] for char in page.chars if 'fontname' in char and char['text'].strip()]
        if len(sizes) < 100: return (10.0, "default")
        body_size = max(set(sizes), key=sizes.count)
        body_name = max(set(names), key=names.count)
        return (body_size, body_name)
    except Exception:
        return (10.0, "default")


def _find_headers_with_pdfplumber(self, page: pdfplumber.page.Page) -> List[ExtractedHeader]:
    """Infers headers by analyzing font sizes.[35, 38]"""
    headers =
    body_size, body_name = self._get_body_font_stats(page)
    size_threshold = body_size * 1.15

    for line in page.lines:
        line_text = line['text'].strip()
        if not line_text: continue

        first_char = line['chars']
        line_size = first_char.get('size', body_size)
        line_name = first_char.get('fontname', body_name)

        is_header = False
        level = 0
        if "bold" in line_name.lower():
            is_header = True
            level = 2
        if line_size > size_threshold:
            is_header = True
            level = 1 if line_size > size_threshold * 1.2 else 2

        if is_header:
            headers.append(ExtractedHeader(
                text=line_text, page_number=page.page_number, level=level,
                font_size=line_size, font_name=line_name, extractor="pdfplumber"
            ))
    return headers


# --- 3.3 Stage 2: Confidence Heuristic ---

def _calculate_pdfplumber_confidence(self, full_text: str) -> float:
    """Calculates confidence for Stage 2."""
    if not full_text.strip():
        return 0.0  # No text extracted, must be scanned.
    return self._check_for_garbled_text(full_text)


# --- 4. Stage 3: AWS Textract Implementation ---

def _textract_extract(self, pdf_path: str) -> ExtractionResult:
    """Stage 3: "Nuclear Option" using AWS Textract.[69]"""
    print(f"Falling back to AWS Textract for {pdf_path}...")
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        response = self.textract_client.analyze_document(
            Document={'Bytes': pdf_bytes},
            FeatureTypes=[12, 39]
        )
        return self._parse_textract_response(response)
    except (ClientError, Exception) as e:
        print(f"AWS Textract failed: {e}")
        return ExtractionResult(
            full_text="", page_texts={}, extractor_used="textract",
            confidence_score=0.0, total_pages=0, raw_response=str(e)
        )


# --- 4.2 Stage 3: Textract Response Parsing ---

def _parse_textract_response(self, response: dict) -> ExtractionResult:
    """Parses the complex Textract JSON response.[45, 47]"""
    all_text =
    page_texts = {}
    tables =
    headers =
    doc = Parser(response)
    total_pages = len(doc.pages)

    for i, page in enumerate(doc.pages):
        page_num = i + 1
        page_text = "\n".join([line.text for line in page.lines])
        all_text.append(page_text)
        page_texts[page_num] = page_text

        for table in page.tables:
            tbl_list = [[cell.text for cell in row.cells] for row in table.rows]
            if not tbl_list: continue
            md_table = self._convert_table_to_markdown(tbl_list)
            tables.append(ExtractedTable(
                page_number=page_num, as_list=tbl_list, as_markdown=md_table,
                confidence=table.confidence, extractor="textract"
            ))

        for field in page.form.fields:
            if field.key and not field.value:
                headers.append(ExtractedHeader(
                    text=field.key.text, page_number=page_num, level=3,
                    font_size=12.0, font_name="N/A", extractor="textract"
                ))

    full_text = "\n\n".join(all_text)
    return ExtractionResult(
        full_text=full_text, page_texts=page_texts, tables=tables,
        headers=headers, extractor_used="textract",
        confidence_score=1.0, total_pages=total_pages, raw_response=response
    )


# --- 0. Main `extract` Method (Public API) ---

def extract(self, pdf_path: str, force_extractor: Optional[str] = None) -> ExtractionResult:
    """
    Runs the full multi-stage extraction pipeline.
    """

    # --- Specialist Router ---
    if self.use_marker and force_extractor is None:
        # Run a lightweight pre-check
        try:
            temp_reader = PdfReader(pdf_path)
            if self._is_technical_doc(pdf_path, temp_reader):
                print(f"Technical document detected. Routing to Marker...")
                marker_result = self._marker_extract(pdf_path)
                if marker_result.confidence_score > 0:
                    return marker_result
                print("Marker failed. Falling back to 3-stage pipeline.")
        except Exception as e:
            print(f"Pre-check failed: {e}. Proceeding with 3-stage pipeline.")

    # Handle forced extractor
    if force_extractor == "marker":
        return self._marker_extract(pdf_path)
    if force_extractor == "pypdf":
        return self._pypdf_extract(pdf_path)
    if force_extractor == "pdfplumber":
        return self._pdfplumber_extract(pdf_path)
    if force_extractor == "textract":
        return self._textract_extract(pdf_path)

    # --- Stage 1: Try fast extraction ---
    result, reader = self._pypdf_extract(pdf_path)
    if result.confidence_score > 0.85:
        print(f"Success: Stage 1 (pypdf) confidence: {result.confidence_score:.2f}")
        return result
    print(f"Info: Stage 1 (pypdf) failed. Confidence: {result.confidence_score:.2f}")

    # --- Stage 2: Try table-aware extraction ---
    # We can re-use the file path
    result = self._pdfplumber_extract(pdf_path)
    if result.confidence_score > 0.75:
        print(f"Success: Stage 2 (pdfplumber) confidence: {result.confidence_score:.2f}")
        return result
    print(f"Info: Stage 2 (pdfplumber) failed. Confidence: {result.confidence_score:.2f}")

    # --- Stage 3: Nuclear option - AWS Textract ---
    print("Falling back to Stage 3 (AWS Textract).")
    return self._textract_extract(pdf_path)


# --- Example Usage ---
if __name__ == "__main__":
    # Create two extractor instances to demonstrate

    # 1. Standard 3-Stage Extractor (no Marker)
    # This is the user's requested implementation
    print("--- Initializing Standard 3-Stage Extractor ---")
    standard_extractor = RobustPDFExtractor(use_marker=False)

    # 2. Specialist Extractor (with Marker)
    # This is the recommended architecture for technical docs
    print("\n--- Initializing Specialist Extractor (with Marker) ---")
    # Set use_marker=True to load the models
    specialist_extractor = RobustPDFExtractor(use_marker=True)

    # --- Example 1: A simple, digitally-born PDF ---
    # (Create a dummy 'simple.pdf' for this to work)
    try:
        # Create a dummy simple PDF
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        # pypdf's text addition is complex; a real file is better.
        # This is a placeholder path.
        SIMPLE_PDF = "simple_test.pdf"
        # Assume 'simple_test.pdf' is a clean, 1-page digital PDF

        print(f"\n--- Processing: {SIMPLE_PDF} (Simple PDF) ---")
        result_simple = standard_extractor.extract(SIMPLE_PDF)
        print(f"Extractor Used: {result_simple.extractor_used}")  # Should be 'pypdf'
        print(f"Confidence: {result_simple.confidence_score}")
        print(f"Text (first 200 chars): {result_simple.full_text[:200]}...")

    except Exception as e:
        print(f"Could not run simple PDF test: {e}")
        print("Please create a 'simple_test.pdf' to run this example.")

    # --- Example 2: A complex, academic PDF (requires Marker) ---
    # (Requires a real PDF, e.g., 'arxiv_paper.pdf')
    try:
        # This is a placeholder path.
        # Download an arXiv paper (e.g., "2308.13418.pdf" for Nougat)
        # to test this path.
        TECH_PDF = "arxiv_paper.pdf"

        print(f"\n--- Processing: {TECH_PDF} (Technical PDF) ---")
        result_tech = specialist_extractor.extract(TECH_PDF)
        print(f"Extractor Used: {result_tech.extractor_used}")  # Should be 'marker'
        print(f"Found {len(result_tech.headers)} headers.")
        print(f"Found {len(result_tech.tables)} tables.")
        print(f"Found {len(result_tech.equations)} equations.")

        if result_tech.equations:
            print("\nSample Equation (LaTeX):")
            print(result_tech.equations.as_latex)

    except Exception as e:
        print(f"Could not run technical PDF test: {e}")
        print(f"Please download an arXiv paper as '{TECH_PDF}' to run this example.")