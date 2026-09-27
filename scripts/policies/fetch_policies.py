"""Fetch retailer policy pages and regulation PDFs into the policy corpus.

Usage (run on a machine with normal internet access):

    python scripts/policies/fetch_policies.py              # fetch every ingest=true source
    python scripts/policies/fetch_policies.py --only amazon-returns-policy
    python scripts/policies/fetch_policies.py --dry-run    # list what would be fetched

Each source is saved as data/policies/<retailer>/<id>.md with its URL and
retrieval date in the front matter. Safety rules:

* A fetch that yields too little text (captcha, login wall, JavaScript-only
  page) is reported and NOT written, so a good existing file is never replaced
  by a broken one. Save such pages by hand using the same header format.
* Files are written atomically (temporary file, then rename).
* An existing file whose text has not changed is left untouched, keeping its
  original retrieval date.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.policy_corpus import (  # noqa: E402
    DEFAULT_POLICY_DIR,
    DEFAULT_SOURCES_FILE,
    MIN_USEFUL_CHARS,
    CorpusError,
    PolicyDocument,
    PolicySource,
    document_path,
    html_to_markdown,
    load_sources,
    parse_document,
    pdf_to_markdown,
    render_document,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


def missing_packages() -> list[str]:
    """Packages the fetcher needs; checked once up front instead of failing per source."""
    import importlib.util

    needed = {"bs4": "beautifulsoup4", "pypdf": "pypdf"}
    return [package for module, package in needed.items() if importlib.util.find_spec(module) is None]


def is_pdf_source(source: PolicySource) -> bool:
    return source.format == "pdf" or source.url.lower().split("?")[0].endswith(".pdf")


def fetch_text(source: PolicySource, timeout: int = 30) -> tuple[str, str | None]:
    response = requests.get(
        source.url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
        timeout=timeout,
    )
    response.raise_for_status()
    is_pdf = source.format == "pdf" or "application/pdf" in response.headers.get("Content-Type", "")
    if is_pdf:
        return pdf_to_markdown(response.content), None
    return html_to_markdown(response.text)


def write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_source(source: PolicySource, body: str, title: str | None, policy_dir: Path) -> str:
    path = document_path(policy_dir, source.retailer, source.id)
    document = PolicyDocument(
        source_id=source.id,
        retailer=source.retailer,
        doc_type=source.doc_type,
        source_url=source.url,
        retrieved_at=date.today().isoformat(),
        body=body,
        extra={"title": title.replace("\n", " ")} if title else {},
    )
    if path.exists():
        try:
            existing = parse_document(path.read_text(encoding="utf-8"), path)
            if existing.content_hash == document.content_hash:
                return "unchanged"
        except CorpusError:
            pass  # replace an unreadable file with a valid one
    write_atomically(path, render_document(document))
    return "saved"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--only", nargs="*", help="Source ids to fetch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)

    missing = missing_packages()
    if missing and not args.dry_run:
        print(
            "Missing Python packages: " + ", ".join(missing)
            + ". Run: pip install -r requirements.txt (in the same environment)."
        )
        return 2

    sources = [source for source in load_sources(args.sources) if source.ingest]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {source.id for source in sources}
        if unknown:
            parser.error(f"Unknown or non-ingest source ids: {', '.join(sorted(unknown))}")
        sources = [source for source in sources if source.id in wanted]

    failures = 0
    for source in sources:
        target = document_path(args.out, source.retailer, source.id)
        if args.dry_run:
            print(f"[dry-run] {source.id:40s} -> {target}")
            continue
        try:
            body, title = fetch_text(source, args.timeout)
        except requests.HTTPError as exc:  # network errors must not stop the batch
            failures += 1
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403, 429}:
                print(
                    f"[blocked] {source.id}: the site refused scripted access (HTTP {status}). "
                    f"Save it by hand to {target} (see data/policies/README.md)."
                )
            elif status in {404, 410}:
                print(
                    f"[gone   ] {source.id}: the page no longer exists (HTTP {status}). "
                    "Find the new URL or set \"ingest\": false in sources.json."
                )
            else:
                print(f"[failed ] {source.id}: {exc}")
            continue
        except Exception as exc:  # network errors must not stop the batch
            failures += 1
            print(f"[failed ] {source.id}: {exc}")
            continue
        if len(body) < MIN_USEFUL_CHARS:
            failures += 1
            if is_pdf_source(source) and not body.strip():
                reason = (
                    "the PDF has no text layer (it is probably a scanned image). "
                    "Copy the text from a text-based copy of the document"
                )
            elif is_pdf_source(source):
                reason = f"the PDF contains only {len(body)} characters of text"
            else:
                reason = (
                    f"only {len(body)} characters of text; the page is probably blocked "
                    "or rendered by JavaScript"
                )
            print(
                f"[too short] {source.id}: {reason}. Save it by hand to {target} "
                "with the same front-matter header (see data/policies/README.md)."
            )
            continue
        status = save_source(source, body, title, args.out)
        print(f"[{status:7s}] {source.id} ({len(body):,} chars) -> {target}")

    if not args.dry_run:
        print(f"\nDone: {len(sources) - failures} ok, {failures} need attention.")
        print("Next: python scripts/policies/build_policy_index.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
