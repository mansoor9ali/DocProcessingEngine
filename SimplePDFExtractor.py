import logging
from typing import Optional, Dict, Any, List
import pdfplumber


class SimplePDFExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text(self, filepath: str) -> Optional[str]:
        """Extract text with automatic fallback"""
        # Try pdfplumber first (primary method)
        result = self._try_pdfplumber(filepath)
        if result:
            self.logger.info(f"Extracted with pdfplumber: {filepath}")
            return result

        # Fallback to PyPDF2
        result = self._try_pypdf2(filepath)
        if result:
            self.logger.warning(f"Fallback to PyPDF2: {filepath}")
            return result

        self.logger.error(f"Both methods failed: {filepath}")
        return None

    def _try_pdfplumber(self, filepath: str) -> Optional[str]:
        """Extract text using pdfplumber."""
        try:
            with pdfplumber.open(filepath) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        except Exception as e:
            self.logger.debug(f"pdfplumber failed: {e}")
            return None

    def _try_pypdf2(self, filepath: str) -> Optional[str]:
        """Extract text using PyPDF2 with basic handling for encrypted PDFs."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            # Attempt to decrypt if necessary (empty password)
            if getattr(reader, 'is_encrypted', False):
                try:
                    reader.decrypt("")
                except Exception:
                    # Continue; some PDFs allow reading text without explicit decrypt
                    pass
            pages = getattr(reader, 'pages', [])
            text = '\n'.join((page.extract_text() or '') for page in pages)
            return text if text is not None else None
        except Exception as e:
            self.logger.debug(f"PyPDF2 failed: {e}")
            return None

    def _extract_tables(self, pdf) -> List[Dict[str, Any]]:
        """Extract tables from an already opened pdfplumber PDF object."""
        all_tables: List[Dict[str, Any]] = []
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables(table_settings={
                # "vertical_strategy": "lines_strict",
                # "horizontal_strategy": "lines_strict",
                "intersection_tolerance": 3,
                "snap_tolerance": 3
            })
            for table_num, table in enumerate(tables or []):
                if table:  # Only add non-empty tables
                    all_tables.append({
                        'page': page_num + 1,
                        'table': table_num + 1,
                        'data': table,
                        'rows': len(table),
                        'cols': len(table[0]) if table and len(table) > 0 else 0
                    })
        return all_tables

    def extract_tables_only(self, filepath: str) -> List[Dict[str, Any]]:
        """Extract only tables from the PDF using pdfplumber.
        Returns an empty list if extraction fails.
        """
        try:
            with pdfplumber.open(filepath) as pdf:
                return self._extract_tables(pdf)
        except Exception as e:
            self.logger.debug(f"pdfplumber (tables only) failed: {e}")
            return []

    def extract(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Extract both text and tables with automatic fallback.
        Returns a dict with keys: text, tables, extractor, page_count.
        """
        # Try pdfplumber first (primary method)
        result = self._try_pdfplumber_with_tables(filepath)
        if result:
            self.logger.info(f"Extracted text and tables with pdfplumber: {filepath}")
            return result

        # Fallback to PyPDF2 for text only
        text = self._try_pypdf2(filepath)
        if text:
            self.logger.warning(f"Fallback to PyPDF2 (text only): {filepath}")
            return {
                'text': text,
                'tables': [],
                'extractor': 'pypdf2',
                'page_count': self._get_pdf_page_count(filepath)
            }

        self.logger.error(f"Both methods failed: {filepath}")
        return None

    def _try_pdfplumber_with_tables(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Extract both text and tables using pdfplumber."""
        try:
            with pdfplumber.open(filepath) as pdf:
                # Extract text
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

                # Extract tables
                tables = self._extract_tables(pdf)

                return {
                    'text': text,
                    'tables': tables,
                    'extractor': 'pdfplumber',
                    'page_count': len(pdf.pages)
                }
        except Exception as e:
            self.logger.debug(f"pdfplumber failed: {e}")
            return None

    def _get_pdf_page_count(self, filepath: str) -> int:
        """Best-effort page count using pdfplumber, falling back to PyPDF2."""
        try:
            with pdfplumber.open(filepath) as pdf:
                return len(pdf.pages)
        except Exception:
            try:
                from PyPDF2 import PdfReader
                return len(PdfReader(filepath).pages)
            except Exception:
                return 0


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    extractor = SimplePDFExtractor()

    # Extract text and tables together
    print("\n=== Complete Extraction (Text + Tables) ===")
    result = extractor.extract('./pdfs/sample-tables.pdf')
    if result:
        print(f"Extractor used: {result['extractor']}")
        print(f"Page count: {result['page_count']}")
        print(f"Tables found: {len(result['tables'])}")
        text_preview = (result['text'] or '')[:500]
        print(f"Text preview (first 500 chars):\n{text_preview}")
        if result['tables']:
            print(f"\nFirst table details:")
            print(f"  Page: {result['tables'][0]['page']}")
            print(f"  Rows: {result['tables'][0]['rows']}")
            print(f"  Cols: {result['tables'][0]['cols']}")
            # Optional: print first table rows
            for row in result['tables'][0]['data'][:5]:
                print(row)
    else:
        print("Failed to extract data")
