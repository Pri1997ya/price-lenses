"""Policy corpus handling for the Eligibility & Safety Agent (Agent 3).

The corpus is a folder of Markdown files, one per source document, each with a
small front-matter header that records where and when the text was retrieved::

    ---
    source_id: amazon-returns-policy
    retailer: amazon
    doc_type: return_policy
    source_url: https://www.amazon.in/gp/help/customer/display.html?nodeId=202111910
    retrieved_at: 2026-09-24
    ---
    # Returns Policy
    ...

Files can be produced by ``scripts/policies/fetch_policies.py`` or saved by hand
(for pages that block scripted access); the indexer treats both identically.
Everything here is pure text processing with no network access, so it is unit
tested directly.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_DIR = REPOSITORY_ROOT / "data" / "policies"
DEFAULT_SOURCES_FILE = DEFAULT_POLICY_DIR / "sources.json"

REQUIRED_FIELDS = ("source_id", "retailer", "doc_type", "source_url", "retrieved_at")

# Marketplace domains used by the Market Investigator -> policy retailer keys.
MARKETPLACE_TO_RETAILER = {
    "amazon.in": "amazon",
    "flipkart.com": "flipkart",
    "croma.com": "croma",
    "reliancedigital.in": "reliance_digital",
    "vijaysales.com": "vijay_sales",
}
RETAILER_LABELS = {
    "amazon": "Amazon",
    "flipkart": "Flipkart",
    "croma": "Croma",
    "reliance_digital": "Reliance Digital",
    "vijay_sales": "Vijay Sales",
    "regulation": "Government rules",
}
REGULATION = "regulation"

# A fetched page with less readable text than this is almost certainly a
# captcha, a login wall, or a JavaScript shell rather than the policy itself.
MIN_USEFUL_CHARS = 400


class CorpusError(ValueError):
    """Raised for malformed corpus files or source lists."""


@dataclass(frozen=True)
class PolicySource:
    id: str
    retailer: str
    doc_type: str
    url: str
    format: str = "html"
    ingest: bool = True
    note: str | None = None


@dataclass
class PolicyDocument:
    source_id: str
    retailer: str
    doc_type: str
    source_url: str
    retrieved_at: str
    body: str
    path: Path | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    text: str
    metadata: dict


def retailer_for_marketplace(marketplace: str | None) -> str | None:
    if not marketplace:
        return None
    host = marketplace.lower().removeprefix("www.")
    for domain, retailer in MARKETPLACE_TO_RETAILER.items():
        if host == domain or host.endswith("." + domain):
            return retailer
    return None


# ---------------------------------------------------------------- sources ----
def load_sources(path: Path = DEFAULT_SOURCES_FILE) -> list[PolicySource]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path} is not valid JSON: {exc}") from exc
    sources = []
    seen: set[str] = set()
    for entry in payload.get("sources", []):
        missing = [key for key in ("id", "retailer", "doc_type", "url") if not entry.get(key)]
        if missing:
            raise CorpusError(f"Source entry {entry!r} is missing {', '.join(missing)}")
        if entry["id"] in seen:
            raise CorpusError(f"Duplicate source id: {entry['id']}")
        seen.add(entry["id"])
        sources.append(
            PolicySource(
                id=entry["id"],
                retailer=entry["retailer"],
                doc_type=entry["doc_type"],
                url=entry["url"],
                format=entry.get("format", "html"),
                ingest=bool(entry.get("ingest", True)),
                note=entry.get("note"),
            )
        )
    return sources


# ----------------------------------------------------------- front matter ----
def parse_document(text: str, path: Path | None = None) -> PolicyDocument:
    """Parse a corpus Markdown file; raises CorpusError if the header is invalid."""
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, flags=re.DOTALL)
    where = f" in {path}" if path else ""
    if not match:
        raise CorpusError(f"Missing front-matter header{where}")
    header: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise CorpusError(f"Malformed header line {line!r}{where}")
        header[key.strip()] = value.strip().strip('"').strip("'")
    missing = [key for key in REQUIRED_FIELDS if not header.get(key)]
    if missing:
        raise CorpusError(f"Header is missing {', '.join(missing)}{where}")
    body = match.group(2).strip()
    if not body:
        raise CorpusError(f"Document body is empty{where}")
    extra = {key: value for key, value in header.items() if key not in REQUIRED_FIELDS}
    return PolicyDocument(
        source_id=header["source_id"],
        retailer=header["retailer"],
        doc_type=header["doc_type"],
        source_url=header["source_url"],
        retrieved_at=header["retrieved_at"],
        body=body,
        path=path,
        extra=extra,
    )


def render_document(document: PolicyDocument) -> str:
    lines = ["---"]
    lines += [f"{key}: {getattr(document, key)}" for key in REQUIRED_FIELDS]
    lines += [f"{key}: {value}" for key, value in sorted(document.extra.items())]
    lines += ["---", "", document.body.strip(), ""]
    return "\n".join(lines)


def document_path(policy_dir: Path, retailer: str, source_id: str) -> Path:
    return Path(policy_dir) / retailer / f"{source_id}.md"


def load_corpus(policy_dir: Path = DEFAULT_POLICY_DIR) -> tuple[list[PolicyDocument], list[str]]:
    """Load every ``*.md`` under the corpus folder. Bad files are reported, not fatal."""
    documents: list[PolicyDocument] = []
    problems: list[str] = []
    seen: dict[str, Path] = {}
    for path in sorted(Path(policy_dir).rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            document = parse_document(path.read_text(encoding="utf-8"), path)
        except (CorpusError, UnicodeDecodeError) as exc:
            problems.append(str(exc))
            continue
        if document.source_id in seen:
            problems.append(
                f"Duplicate source_id {document.source_id} in {path} (already in {seen[document.source_id]}); skipped"
            )
            continue
        seen[document.source_id] = path
        documents.append(document)
    return documents, problems


# --------------------------------------------------------- text extraction ----
_DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form", "button", "header", "footer", "nav")


def _flatten_table_rows(soup, root) -> None:
    """Turn each table row into one line: "Mobiles | 7 days Replacement only | ...".

    Retailer return policies are mostly tables of category -> window ->
    conditions. Emitting cells separately scatters a rule across lines (and
    passages), so no passage says "Mobiles: 7 days replacement only".
    """
    for row in root.find_all("tr"):
        if row.find("table"):
            continue  # nested table: its own rows are flattened instead
        cells = [
            re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
            for cell in row.find_all(["td", "th"], recursive=False)
        ]
        cells = [cell for cell in cells if cell]
        if not cells:
            row.decompose()
            continue
        item = soup.new_tag("li")
        item.string = " | ".join(cells)
        row.replace_with(item)


def html_to_markdown(html: str) -> tuple[str, str | None]:
    """Convert a policy page to heading-preserving plain Markdown. Returns (text, title)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    _flatten_table_rows(soup, root)
    lines: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd"]):
        # Skip containers whose text is already emitted through a nested block element.
        if element.name in {"li", "td", "dd"} and element.find(["p", "li", "table"]):
            continue
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if element.name.startswith("h"):
            level = min(int(element.name[1]), 4)
            lines.append("")
            lines.append(f"{'#' * level} {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    if not any(line.strip() for line in lines):
        text = re.sub(r"\s+\n", "\n", root.get_text("\n", strip=True))
        lines = [text]
    collapsed: list[str] = []
    for line in lines:
        if collapsed and line == collapsed[-1] and line:
            continue  # repeated menu items / duplicated responsive markup
        collapsed.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(collapsed)).strip(), title


def pdf_to_markdown(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## Page {number}\n\n{text}")
    return "\n\n".join(pages).strip()


# ---------------------------------------------------------------- chunking ----
def _split_sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in body.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
        if match:
            if any(part.strip() for part in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)
    if any(part.strip() for part in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=- )", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > size:  # a single very long paragraph
            cut = paragraph.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[max(cut - overlap, 0):].strip()
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_document(document: PolicyDocument, size: int = 1000, overlap: int = 150) -> list[PolicyChunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[PolicyChunk] = []
    for heading, text in _split_sections(document.body):
        for piece in _window(text, size, overlap):
            index = len(chunks)
            prefix = f"{heading}\n" if heading else ""
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{document.source_id}::{index:04d}",
                    text=f"{prefix}{piece}".strip(),
                    metadata={
                        "source_id": document.source_id,
                        "retailer": document.retailer,
                        "doc_type": document.doc_type,
                        "source_url": document.source_url,
                        "retrieved_at": document.retrieved_at,
                        "heading": heading,
                        "chunk_index": index,
                        "content_hash": document.content_hash,
                    },
                )
            )
    return chunks
