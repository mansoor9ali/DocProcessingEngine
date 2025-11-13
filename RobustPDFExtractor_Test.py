import re
import boto3
import nltk
import pdfplumber
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from gibberish_detector import detector
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


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
    # Placeholder for image rendering if needed
    as_image: Optional[bytes] = None
    extractor: str = ""


@dataclass
class ExtractionResult:
    """
      A unified data structure to hold the output of any extraction stage.
      This object is the "product" of the extraction pipeline and the
      "input" to the RAG chunking/embedding pipeline.
      """
    # Required fields first (no defaults)
    full_text: str
    page_texts: Dict[int, str]
    extractor_used: str
    confidence_score: float
    total_pages: int

    # Optional fields with defaults last
    tables: List[ExtractedTable] = field(default_factory=list)
    headers: List[ExtractedHeader] = field(default_factory=list)
    equations: List[ExtractedEquation] = field(default_factory=list)
    raw_response: Optional[Any] = None


# --- 2. Stage 1: pypdf Implementation ---

class RobustPDFExtractor_Test:
    def __init__(self, aws_region_name: str = 'us-east-1'):
        """
        Initializes the extractor, setting up clients and heuristics.
        """
        self.textract_client = boto3.client('textract', region_name=aws_region_name)

        # Initialize heuristics for Stage 1
        self._initialize_stage1_heuristics()

    def _initialize_stage1_heuristics(self):
        """
        Pre-loads resources needed for confidence scoring.
        """
        try:
            nltk.data.find('corpus/words')
        except LookupError:
            print("NLTK 'words' corpus not found. Downloading...")
            nltk.download('words')

        self.english_words = set(nltk.corpus.words.words())

        # Initialize the gibberish detector - use create() factory method
        try:
            self.gibberish_detector = detector.create_from_model('madhurjindal/autonlp-Gibberish-Detector-492513457')
        except Exception as e:
            print(f"Warning: gibberish_detector initialization failed: {e}")
            self.gibberish_detector = None

    def _pypdf_extract(self, pdf_path: str) -> ExtractionResult:
        """
        Stage 1: Fast text extraction using pypdf.
        """
        all_text = []
        page_texts = {}
        total_pages = 0

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                page_num = i + 1
                try:
                    # Use standard text extraction
                    text = page.extract_text()
                    if text is None:
                        text = ""
                except Exception as e:
                    # Handle potential extraction errors on a page
                    print(f"pypdf error on page {page_num}: {e}")
                    text = ""

                all_text.append(text)
                page_texts[page_num] = text

            full_text = "\n\n".join(all_text)

            # Calculate the confidence score for this extraction
            confidence = self._calculate_pypdf_confidence(
                reader=reader,
                full_text=full_text,
                page_texts=page_texts
            )

            return ExtractionResult(
                full_text=full_text,
                page_texts=page_texts,
                extractor_used="pypdf",
                confidence_score=confidence,
                total_pages=total_pages
            )

        except PdfReadError as e:
            # This indicates a corrupted or unreadable PDF
            print(f"pypdf failed to read file: {e}")
            return ExtractionResult(
                full_text="",
                page_texts={},
                extractor_used="pypdf",
                confidence_score=0.0,  # Definitely failed
                total_pages=0
            )
        except Exception as e:
            print(f"Unhandled pypdf error: {e}")
            return ExtractionResult(
                full_text="",
                page_texts={},
                extractor_used="pypdf",
                confidence_score=0.0,
                total_pages=0
            )

    @staticmethod
    def _check_for_scanned_pages(reader: PdfReader, page_texts: Dict[int, str]) -> float:
        """
        Detects scanned pages.
        Heuristic: A page is likely scanned if text extraction yields
        very little text, but the page *does* contain image objects.
        """
        scanned_page_count = 0
        total_pages = len(reader.pages)
        if total_pages == 0:
            return 0.0  # No pages, no confidence

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text_length = len(page_texts.get(page_num, "").strip())

            # pypdf's.images property lists images on the page
            has_images = len(page.images) > 0

            # A page is considered "scanned" if it has almost no text
            # but *does* have images.
            if text_length < 50 and has_images:
                # This is a strong signal for a scanned page
                scanned_page_count += 1
            elif text_length < 50 and not has_images:
                # This might just be a blank or title page, which is fine.
                pass

        # The penalty is proportional to the percentage of scanned pages.
        scanned_ratio = scanned_page_count / total_pages

        # Confidence is 1.0 (good) minus the scanned ratio.
        # If 50% of pages are scanned, confidence drops by 0.5.
        return 1.0 - scanned_ratio

    def _check_for_garbled_text(self, full_text: str) -> float:
        """
        Detects garbled or "mojibake" text from encoding errors.
        Heuristic: Use a pre-trained gibberish detector.
        """
        # Clean the text for a fair check
        text_to_check = re.sub(r'\s+', ' ', full_text).strip()

        if len(text_to_check) < 100:
            return 1.0  # Not enough text to make a decision, assume OK.

        # self.gibberish_detector.is_gibberish() returns True if it's gibberish.
        if self.gibberish_detector and self.gibberish_detector.is_gibberish(text_to_check):
            return 0.0  # High confidence this is garbage.

        # Fallback check using NLTK word list
        words = set(text_to_check.lower().split())
        if not words:
            return 1.0  # Empty is not garbled.

        intersection = words.intersection(self.english_words)
        word_ratio = len(intersection) / len(words)

        if word_ratio < 0.3:
            return 0.3  # Low ratio of real words, low confidence.

        return 1.0  # Looks like real text.

    @staticmethod
    def _check_for_layout_collapse(page_texts: Dict[int, str]) -> float:
        """
        Detects multi-column layout collapse.
        Heuristic: Collapsed layouts have very long lines
        (low newline-to-char ratio).
        """
        total_chars = 0
        total_newlines = 0

        for text in page_texts.values():
            total_chars += len(text)
            total_newlines += text.count('\n')

        if total_chars == 0 or total_newlines == 0:
            return 1.0  # No text, or single line (e.g., title page), fine.

        chars_per_newline = total_chars / total_newlines

        # This threshold is heuristic. A normal page of text
        # might have 80-120 chars per line. A collapsed
        # multi-column layout will have 500+ chars per line.
        if chars_per_newline > 300:
            return 0.5  # Suspect layout collapse.

        return 1.0

    def _calculate_pypdf_confidence(self, reader: PdfReader, full_text: str, page_texts: Dict[int, str]) -> float:
        """
        Combines all heuristics into a single confidence score.
        We use a weighted average, where "scanned" and "garbled"
        are the most important signals.
        """
        scanned_score = self._check_for_scanned_pages(reader, page_texts)
        garbled_score = self._check_for_garbled_text(full_text)
        layout_score = self._check_for_layout_collapse(page_texts)

        # If text is totally garbled, confidence is 0.
        if garbled_score == 0.0:
            return 0.0

        # Weighted average. Scanned detection is most important.
        final_score = (
                (scanned_score * 0.6) +
                (garbled_score * 0.3) +
                (layout_score * 0.1)
        )

        return final_score

    def _convert_table_to_markdown(self, table_data: List[List[Optional[str]]]) -> str:
        """
        Converts a list-of-lists table into a Markdown string.
        """
        if not table_data:
            return ""

        # Create header row
        header = "| " + " | ".join(str(h) for h in table_data) + " |"
        # Create separator row
        separator = "| " + " | ".join(["---"] * len(table_data)) + " |"
        # Create data rows
        rows = [
            "| " + " | ".join(str(cell) if cell is not None else "" for cell in row) + " |"
            for row in table_data[1:]]

        return "\n".join([header, separator] + rows)

    def _calculate_pdfplumber_confidence(self, full_text: str) -> float:
        """
        Calculates confidence for Stage 2.
        The main check is: did we get non-garbled text?
        """
        if not full_text.strip():
            return 0.0  # No text extracted, definitely a scanned PDF.

        # Re-use the Stage 1 garbled text detector
        garbled_score = self._check_for_garbled_text(full_text)

        return garbled_score

    def _pdfplumber_extract(self, pdf_path: str) -> ExtractionResult:
        """
        Stage 2: Structure-aware extraction using pdfplumber.
        """
        all_text = []
        page_texts = {}
        tables = []
        headers = []
        total_pages = 0

        # These are settings for "borderless" tables [32, 33]
        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 4,
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1

                    # 1. Extract layout-aware text
                    # The x_tolerance helps respect column boundaries [35]
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text is None:
                        text = ""

                    all_text.append(text)
                    page_texts[page_num] = text

                    # 2. Extract tables
                    extracted_tables = page.extract_tables(table_settings=table_settings)
                    for tbl_list in extracted_tables:
                        if not tbl_list:
                            continue

                        md_table = self._convert_table_to_markdown(tbl_list)
                        tables.append(ExtractedTable(
                            page_number=page_num,
                            as_list=tbl_list,
                            as_markdown=md_table,
                            extractor="pdfplumber"
                        ))

                    # 3. Extract Headers
                    headers.extend(
                        self._find_headers_with_pdfplumber(page)
                    )

            full_text = "\n\n".join(all_text)

            # Calculate confidence for this stage
            confidence = self._calculate_pdfplumber_confidence(
                full_text=full_text
            )

            return ExtractionResult(
                full_text=full_text,
                page_texts=page_texts,
                tables=tables,
                headers=headers,
                extractor_used="pdfplumber",
                confidence_score=confidence,
                total_pages=total_pages
            )
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            return ExtractionResult(
                full_text="",
                page_texts={},
                extractor_used="pdfplumber",
                confidence_score=0.0,  # Failed, move to Stage 3
                total_pages=0
            )

    def _get_body_font_stats(self, page: pdfplumber.page.Page, min_chars: int = 100) -> (float, str):
        """
        Helper to find the most common (body) font size and name.
        """
        try:
            # Get font sizes and names from character data
            sizes = [
                char['size'] for char in page.chars
                if 'size' in char and char['text'].strip()
            ]
            names = [
                char['fontname'] for char in page.chars
                if 'fontname' in char and char['text'].strip()
            ]

            if len(sizes) < min_chars:
                return (10.0, "default")  # Not enough data, return a sensible default

            # Find the most common font size (mode)
            body_size = max(set(sizes), key=sizes.count)
            # Find the most common font name (mode)
            body_name = max(set(names), key=names.count)

            return (body_size, body_name)
        except Exception:
            return (10.0, "default")  # Fallback

    def _find_headers_with_pdfplumber(self, page: pdfplumber.page.Page) -> List[ExtractedHeader]:
        """
        Infers headers by analyzing font sizes.
        Heuristic: Headers are lines with font size > body font size,
        or a "Bold" font name.[35, 37, 38]
        """
        headers = []
        body_size, body_name = self._get_body_font_stats(page)

        # Define thresholds
        size_threshold = body_size * 1.15  # e.g., 11.5pt if body is 10pt

        # group_lines combines chars into lines
        for line in page.lines:
            line_text = line['text'].strip()
            if not line_text:
                continue

            # Get the font size/name of the *first char* in the line
            first_char = line['chars']
            line_size = first_char.get('size', body_size)
            line_name = first_char.get('fontname', body_name)

            is_header = False
            level = 0

            # Heuristic 1: Font name contains "Bold"
            if "bold" in line_name.lower():
                is_header = True
                level = 2  # Assume bold is H2

            # Heuristic 2: Font size is significantly larger than body
            if line_size > size_threshold:
                is_header = True
                if line_size > size_threshold * 1.2:  # e.g., > 13.8pt
                    level = 1  # H1
                else:
                    level = 2  # H2

            if is_header:
                headers.append(ExtractedHeader(
                    text=line_text,
                    page_number=page.page_number,
                    level=level,
                    font_size=line_size,
                    font_name=line_name,
                    extractor="pdfplumber"
                ))

        return headers

    def _textract_extract(self, pdf_path: str) -> ExtractionResult:
        """
        Stage 3: "Nuclear Option" using AWS Textract for OCR and
        structural analysis.
        """
        print(f"Falling back to AWS Textract for {pdf_path}...")
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            # Use analyze_document for structural data [12, 40]
            response = self.textract_client.analyze_document(
                Document={'Bytes': pdf_bytes},
                FeatureTypes=  # Essential! [39]
            )

            # The Textract JSON response is complex [22, 44]
            # We must parse it into our standard ExtractionResult.
            return self._parse_textract_response(response)

        except ClientError as e:
            print(f"AWS Textract ClientError: {e}")
            return ExtractionResult(
                full_text="",
                page_texts={},
                extractor_used="textract",
                confidence_score=0.0,  # Failed
                total_pages=0,
                raw_response=str(e)
            )
        except Exception as e:
            print(f"Unhandled Textract error: {e}")
            return ExtractionResult(
                full_text="",
                page_texts={},
                extractor_used="textract",
                confidence_score=0.0,
                total_pages=0,
                raw_response=str(e)
            )

    def _parse_textract_response(self, response: dict) -> ExtractionResult:
        """
        Parses the complex Textract JSON response into our
        standard ExtractionResult using the official parser library.[45, 47]
        """
        all_text = []
        page_texts = {}
        tables = []
        headers = []

        doc = Parser(response)  # Use the parser library
        total_pages = len(doc.pages)

        for i, page in enumerate(doc.pages):
            page_num = i + 1

            # 1. Extract Text
            # We join lines to reconstruct page text
            page_text = "\n".join([line.text for line in page.lines])
            all_text.append(page_text)
            page_texts[page_num] = page_text

            # 2. Extract Tables
            # The parser library reconstructs tables [47]
            for table in page.tables:
                # Convert Textract's table object to list-of-lists
                tbl_list =
                for r_idx, row in enumerate(table.rows):
                    row_list =
                    for c_idx, cell in enumerate(row.cells):
                        row_list.append(cell.text)
                    tbl_list.append(row_list)

                if not tbl_list:
                    continue

                md_table = self._convert_table_to_markdown(tbl_list)
                tables.append(ExtractedTable(
                    page_number=page_num,
                    as_list=tbl_list,
                    as_markdown=md_table,
                    confidence=table.confidence,
                    extractor="textract"
                ))

            # 3. Extract "Headers" (from FORMS)
            # We can infer headers from Form Key-Value pairs [12]
            for field in page.form.fields:
                if field.key and not field.value:
                    # If a "key" has no "value", it's often a title or header
                    headers.append(ExtractedHeader(
                        text=field.key.text,
                        page_number=page_num,
                        level=3,  # Assume H3 for form keys
                        font_size=12.0,  # Not available, use default
                        font_name="N/A",  # Not available
                        extractor="textract"
                    ))

        full_text = "\n\n".join(all_text)

        # Stage 3 is the end of the line. Confidence is always 1.0.
        # This is the "ground truth" for this pipeline.
        return ExtractionResult(
            full_text=full_text,
            page_texts=page_texts,
            tables=tables,
            headers=headers,
            extractor_used="textract",
            confidence_score=1.0,
            total_pages=total_pages,
            raw_response=response
        )


# --- Example Usage ---
if __name__ == "__main__":
    # Create two extractor instances to demonstrate

    # 1. Standard 3-Stage Extractor (no Marker)
    # This is the user's requested implementation
    print("--- Initializing Standard 3-Stage Extractor ---")
    standard_extractor = RobustPDFExtractor_Test(use_marker=False)

    # 2. Specialist Extractor (with Marker)
    # This is the recommended architecture for technical docs
    print("\n--- Initializing Specialist Extractor (with Marker) ---")
    # Set use_marker=True to load the models
    specialist_extractor = RobustPDFExtractor_Test(use_marker=True)

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

