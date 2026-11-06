from io import BytesIO
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

def extract_text(filename: str, data: bytes) -> str:
    low = filename.lower()
    if low.endswith('.pdf'):
        r = PdfReader(BytesIO(data))
        return '\n'.join((p.extract_text() or '') for p in r.pages)
    if low.endswith('.docx'):
        d = Document(BytesIO(data))
        return '\n'.join(p.text for p in d.paragraphs)
    if low.endswith('.xlsx'):
        wb = load_workbook(BytesIO(data), read_only=True, data_only=True)
        lines=[]
        for ws in wb.worksheets:
            lines.append(f"# {ws.title}")
            for row in ws.iter_rows(values_only=True):
                lines.append(' | '.join('' if v is None else str(v) for v in row))
        return '\n'.join(lines)
    return data.decode('utf-8', errors='ignore')

def chunks(text: str, size=1800, overlap=250):
    text = ' '.join(text.split())
    i=0
    while i < len(text):
        yield text[i:i+size]
        i += max(1, size-overlap)
