import logging
from typing import Optional, Dict, Any
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
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        except Exception as e:
            self.logger.debug(f"pdfplumber failed: {e}")
            return None

    def _try_pypdf2(self, filepath) -> Optional[str]:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
        except Exception as e:
            self.logger.debug(f"PyPDF2 failed: {e}")
            return None

    def _extract_tables(self, pdf) -> list:
        """Extract tables from an already opened pdfplumber PDF object"""
        all_tables = []
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables(table_settings={
                "intersection_tolerance": 3,
                "snap_tolerance": 3
            })
            for table_num, table in enumerate(tables):
                if table:  # Only add non-empty tables
                    all_tables.append({
                        'page': page_num + 1,
                        'table': table_num + 1,
                        'data': table,
                        'rows': len(table),
                        'cols': len(table[0]) if table else 0
                    })
        return all_tables

    def extract(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Extract both text and tables with automatic fallback"""
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
                'page_count': 0
            }

        self.logger.error(f"Both methods failed: {filepath}")
        return None

    def _try_pdfplumber_with_tables(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Extract both text and tables using pdfplumber"""
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

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    extractor = SimplePDFExtractor()

    # Extract text and tables together
    print("\n=== Complete Extraction (Text + Tables) ===")
    result = extractor.extract('./pdfs/sample-tables2.pdf')
    if result:
        print(f"Extractor used: {result['extractor']}")
        print(f"Page count: {result['page_count']}")
        print(f"Tables found: {len(result['tables'])}")
        print(f"Text preview: {result['text']}")
        if result['tables']:
            print(f"\nFirst table details:")
            print(f"  Page: {result['tables'][0]['page']}")
            print(f"  Rows: {result['tables'][0]['rows']}")
            print(f"  Cols: {result['tables'][0]['cols']}")
            for table_info in result['tables']:
                print(
                    f"Page: {table_info['page']}, Table: {table_info['table']}, Rows: {table_info['rows']}, Cols: {table_info['cols']}")
                for row in table_info['data']:
                    print(row)
                print("\n")
    else:
        print("Failed to extract data")

