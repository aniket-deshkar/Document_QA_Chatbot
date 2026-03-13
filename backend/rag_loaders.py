"""Document loaders for text, tables, OCR images, PDFs, and presentations."""

import importlib
import os
import re
import shutil
import subprocess
import tempfile
from glob import glob
from typing import List

import pandas as pd
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter

from .config import settings

TABLE_EXTENSIONS = {".csv", ".xls", ".xlsx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}
PRESENTATION_EXTENSIONS = {".ppt", ".pptx"}


def _find_table_column(columns: List[str], candidates: tuple[str, ...]) -> str | None:
    """Find a likely table column by case-insensitive name match."""
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        col = lower_map.get(name.lower())
        if col:
            return col
    return None


def _extract_year_series(series: pd.Series) -> pd.Series:
    """Extract 4-digit years from date/period strings."""
    return series.astype(str).str.extract(r"((?:19|20)\d{2})", expand=False).fillna("")


def _sample_table_rows(df: pd.DataFrame, max_rows: int, period_col: str | None) -> pd.DataFrame:
    """Downsample large tables while preserving year coverage where possible."""
    if len(df) <= max_rows:
        return df
    if not period_col:
        return df.head(max_rows).copy()

    years = _extract_year_series(df[period_col])
    year_values = sorted([y for y in years.unique().tolist() if y])
    if not year_values:
        return df.head(max_rows).copy()

    per_year_quota = max(1, max_rows // max(1, len(year_values)))
    selected_idx: list[int] = []
    for year in year_values:
        year_idx = df.index[years == year].tolist()
        selected_idx.extend(year_idx[:per_year_quota])

    if len(selected_idx) < max_rows:
        selected_set = set(selected_idx)
        for idx in df.index.tolist():
            if idx in selected_set:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= max_rows:
                break

    selected_idx = sorted(selected_idx[:max_rows])
    return df.loc[selected_idx].copy()


def _build_row_oriented_chunk(df_chunk: pd.DataFrame, columns: List[str], base_row: int) -> str:
    """Render tabular rows as key=value lines for stronger retrieval signals."""
    lines: list[str] = []
    for offset, row in enumerate(df_chunk.itertuples(index=False), start=0):
        parts: list[str] = []
        for col, value in zip(columns, row):
            value_str = str(value).strip()
            if not value_str:
                continue
            parts.append(f"{col}={value_str}")
        if parts:
            lines.append(f"Row {base_row + offset}: " + " | ".join(parts))
    return "\n".join(lines)


def _get_ocr_modules():
    """Import OCR dependencies and ensure Tesseract binary is reachable."""
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except Exception as e:
        raise RuntimeError(
            "OCR dependencies missing. Install `pytesseract` and ensure `tesseract` binary is installed."
        ) from e

    configured_cmd = (settings.TESSERACT_CMD or "").strip()
    candidates = [
        configured_cmd,
        shutil.which("tesseract") or "",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for cmd in candidates:
        if cmd and os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
            break

    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract binary is not usable. Set TESSERACT_CMD in .env to the absolute tesseract path."
        ) from e

    return pytesseract, Image, ImageOps


def _build_ocr_image_variants(image, ImageOps) -> list:
    """Generate preprocessed image variants to improve OCR robustness."""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    width, height = normalized.size
    if min(width, height) < 1200:
        scale = max(1.0, 1200.0 / max(1, min(width, height)))
        normalized = normalized.resize(
            (int(width * scale), int(height * scale)),
            resample=3,  # bicubic
        )

    gray = ImageOps.grayscale(normalized)
    contrast = ImageOps.autocontrast(gray)
    binary = contrast.point(lambda p: 255 if p > 165 else 0)
    return [normalized, contrast, binary]


def _ocr_score(text: str) -> int:
    """Score OCR text candidates to select the most informative extraction."""
    if not text:
        return 0
    clean = text.strip()
    alnum = len(re.findall(r"[A-Za-z0-9]", clean))
    lines = len([ln for ln in clean.splitlines() if ln.strip()])
    return len(clean) + (2 * alnum) + (5 * lines)


def _extract_text_with_ocr(pytesseract, image, ImageOps) -> str:
    """Run OCR across variants/configurations and return the best text result."""
    variants = _build_ocr_image_variants(image, ImageOps)
    configs = (
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 3",
    )

    best_text = ""
    best_score = -1
    for variant in variants:
        for config in configs:
            try:
                text = pytesseract.image_to_string(variant, config=config) or ""
            except Exception:
                continue
            score = _ocr_score(text)
            if score > best_score:
                best_score = score
                best_text = text
    return best_text.strip()


def _load_table_documents(file_path: str) -> List[Document]:
    """Load CSV/Excel files into schema and row-chunk documents."""
    ext = os.path.splitext(file_path.lower())[1]
    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=False,
                low_memory=False,
                encoding="latin-1",
            )
    else:
        df = pd.read_excel(file_path, dtype=str)

    df = df.fillna("")
    source = os.path.basename(file_path)
    original_total_rows = len(df)
    columns = df.columns.astype(str).tolist()
    period_col = _find_table_column(columns, ("Period", "Date", "Year", "Month"))
    value_col = _find_table_column(columns, ("Data_value", "Value", "Amount", "transaction_value"))
    if original_total_rows > settings.TABLE_MAX_ROWS:
        df = _sample_table_rows(df, settings.TABLE_MAX_ROWS, period_col)

    if df.empty:
        return [Document(text=f"Source file: {source}\nTable is empty.", metadata={"source": source})]

    docs: List[Document] = []
    cols = ", ".join(columns)
    truncation_note = ""
    if original_total_rows > settings.TABLE_MAX_ROWS:
        truncation_note = (
            f"\nNOTE: Table truncated for indexing from {original_total_rows} rows "
            f"to {settings.TABLE_MAX_ROWS} sampled rows for performance and year coverage."
        )
    docs.append(
        Document(
            text=(
                f"Source file: {source}\n"
                f"Columns: {cols}\n"
                f"Rows indexed: {len(df)} of {original_total_rows}"
                f"{truncation_note}"
            ),
            metadata={"source": source, "type": "table_schema"},
        )
    )

    if period_col:
        years = sorted([y for y in _extract_year_series(df[period_col]).unique().tolist() if y])
        if years:
            docs.append(
                Document(
                    text=(
                        f"Source file: {source}\n"
                        f"Detected period column: {period_col}\n"
                        f"Available years in indexed rows: {', '.join(years)}"
                    ),
                    metadata={"source": source, "type": "table_profile", "period_column": period_col},
                )
            )

    if period_col and value_col:
        year_frame = pd.DataFrame(
            {
                "year": _extract_year_series(df[period_col]),
                "value": pd.to_numeric(
                    df[value_col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ),
            }
        )
        year_groups = (
            year_frame[year_frame["year"] != ""]
            .groupby("year")["value"]
            .agg(["sum", "count"])
            .reset_index()
        )
        for row in year_groups.itertuples(index=False):
            year = str(getattr(row, "year", "")).strip()
            if not year:
                continue
            total_value = float(getattr(row, "sum", 0.0) or 0.0)
            count_value = int(getattr(row, "count", 0) or 0)
            docs.append(
                Document(
                    text=(
                        f"Source file: {source}\n"
                        f"Year: {year}\n"
                        f"Value column: {value_col}\n"
                        f"Total value: {total_value:,.2f}\n"
                        f"Numeric rows counted: {count_value}"
                    ),
                    metadata={
                        "source": source,
                        "type": "table_year_aggregate",
                        "year": year,
                        "value_column": value_col,
                    },
                )
            )

    total = len(df)
    for start in range(0, total, settings.TABLE_ROWS_PER_CHUNK):
        end = min(start + settings.TABLE_ROWS_PER_CHUNK, total)
        chunk_df = df.iloc[start:end]
        chunk_text = _build_row_oriented_chunk(chunk_df, columns, start + 1)
        metadata = {"source": source, "type": "table", "row_start": start + 1, "row_end": end}
        if period_col:
            chunk_years = sorted(
                [y for y in _extract_year_series(chunk_df[period_col]).unique().tolist() if y]
            )
            if chunk_years:
                metadata["years"] = ",".join(chunk_years[:8])
        docs.append(
            Document(
                text=f"Source file: {source}\nRows: {start + 1}-{end}\n{chunk_text}",
                metadata=metadata,
            )
        )
    return docs


def _load_image_ocr_documents(file_path: str) -> List[Document]:
    """Extract OCR text from a standalone image file."""
    pytesseract, Image, ImageOps = _get_ocr_modules()
    with Image.open(file_path) as image:
        text = _extract_text_with_ocr(pytesseract, image, ImageOps)

    if not text.strip():
        raise RuntimeError("OCR extracted empty text from image.")
    return [Document(text=text, metadata={"source": os.path.basename(file_path), "type": "image_ocr"})]


def _load_generic_documents(file_path: str) -> List[Document]:
    """Load non-specialized files through LlamaIndex directory reader."""
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    if not documents:
        raise RuntimeError("No readable content found in file.")
    return documents


def _load_pdf_documents(file_path: str) -> List[Document]:
    """Load PDF text directly, with OCR fallback for scanned pages."""
    source = os.path.basename(file_path)

    try:
        docs = _load_generic_documents(file_path)
        combined = "\n".join(doc.text for doc in docs if getattr(doc, "text", ""))
        if len(combined.strip()) >= settings.PDF_TEXT_MIN_LENGTH_FOR_NO_OCR:
            return docs
    except Exception:
        pass

    pytesseract, Image, ImageOps = _get_ocr_modules()
    if not shutil.which("pdftoppm"):
        raise RuntimeError("PDF OCR fallback requires `pdftoppm` (poppler-utils) and `tesseract`.")

    ocr_docs: List[Document] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = os.path.join(tmp_dir, "page")
        try:
            subprocess.run(
                ["pdftoppm", "-r", str(settings.OCR_IMAGE_DPI), "-png", file_path, prefix],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed converting PDF to images for OCR: {e.stderr.decode('utf-8', errors='ignore')}"
            ) from e

        for page_num, image_path in enumerate(sorted(glob(f"{prefix}-*.png")), start=1):
            with Image.open(image_path) as image:
                text = _extract_text_with_ocr(pytesseract, image, ImageOps)
            if text:
                ocr_docs.append(
                    Document(
                        text=f"[Page {page_num}]\n{text}",
                        metadata={"source": source, "page": page_num, "type": "pdf_ocr"},
                    )
                )

    if not ocr_docs:
        raise RuntimeError("No extractable text found in PDF. Ensure Tesseract works for scanned documents.")
    return ocr_docs


def _load_presentation_documents(file_path: str) -> List[Document]:
    """Load PPT/PPTX content slide-wise with generic fallback when needed."""
    ext = os.path.splitext(file_path.lower())[1]
    source = os.path.basename(file_path)

    if ext != ".pptx":
        return _load_generic_documents(file_path)

    try:
        Presentation = getattr(importlib.import_module("pptx"), "Presentation")
    except Exception:
        return _load_generic_documents(file_path)

    docs: List[Document] = []
    prs = Presentation(file_path)
    for slide_idx, slide in enumerate(prs.slides, start=1):
        texts = [
            shape.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and getattr(shape, "text", "")
        ]
        slide_text = "\n".join(text for text in texts if text)
        if slide_text:
            docs.append(
                Document(
                    text=f"[Slide {slide_idx}]\n{slide_text}",
                    metadata={"source": source, "slide": slide_idx, "type": "pptx"},
                )
            )

    return docs or _load_generic_documents(file_path)


def load_documents(file_path: str) -> List[Document]:
    """Dispatch file loading to the appropriate specialized loader."""
    ext = os.path.splitext(file_path.lower())[1]
    if ext in TABLE_EXTENSIONS:
        return _load_table_documents(file_path)
    if ext in IMAGE_EXTENSIONS:
        return _load_image_ocr_documents(file_path)
    if ext in PDF_EXTENSIONS:
        return _load_pdf_documents(file_path)
    if ext in PRESENTATION_EXTENSIONS:
        return _load_presentation_documents(file_path)
    return _load_generic_documents(file_path)


def build_splitter(file_path: str) -> TokenTextSplitter:
    """Return chunk splitter tuned for tabular or plain document content."""
    ext = os.path.splitext(file_path.lower())[1]
    if ext in TABLE_EXTENSIONS:
        return TokenTextSplitter(
            chunk_size=settings.TABLE_TOKEN_CHUNK_SIZE,
            chunk_overlap=0,
            separator="\n",
        )
    return TokenTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separator="\n",
    )
