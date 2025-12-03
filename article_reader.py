# article_reader.py
import fitz  # PyMuPDF

def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_article(path):
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    else:
        return read_text(path)
