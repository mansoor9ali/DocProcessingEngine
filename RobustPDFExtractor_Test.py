import re
import boto3
import nltk
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

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    extractor = RobustPDFExtractor_Test()

    # Extract text and tables together
    print("\n=== Complete Extraction (Text + Tables) ===")
    extractionResult = extractor._pypdf_extract('./pdfs/sample-tables2.pdf')
    print(extractionResult)

