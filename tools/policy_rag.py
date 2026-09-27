"""Chroma-backed policy RAG for the Eligibility & Safety Agent (Agent 3).

Flow: corpus Markdown -> chunks -> embeddings -> Chroma ``retailer_policies``
collection -> retrieval (optionally filtered by retailer) -> grounded answer
with numbered citations.

Grounding rules:
* The LLM may answer only from retrieved passages and must cite them as [n].
* An answer that cites a passage number that was not supplied, or cites
  nothing, is rejected and replaced by the retrieved passages themselves.
* When the LLM is unreachable the retrieved passages are returned directly
  (``mode="extractive"``), so the feature keeps working offline.
* When nothing relevant is retrieved the answer says so (``mode="no_match"``).

Embeddings are computed here and passed to Chroma explicitly, and the backend
name is stored on the collection. Querying with a different backend than the
one used to build the index is refused instead of returning garbage.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .policy_corpus import (
    REGULATION,
    REPOSITORY_ROOT,
    PolicyDocument,
    chunk_document,
)

log = logging.getLogger(__name__)

COLLECTION_NAME = "retailer_policies"
BUILD_COLLECTION_NAME = "retailer_policies_build"
DEFAULT_CHROMA_DIR = REPOSITORY_ROOT / "data" / "chroma"
DEFAULT_MIN_RELEVANCE = 0.25


class PolicyIndexError(RuntimeError):
    """Raised when the policy index is missing or incompatible."""


# -------------------------------------------------------------- embedders ----
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX). Downloads the model once, no API key."""

    name = "local-minilm-l6-v2"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._function = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vector)) for vector in self._function(texts)]


class GatewayEmbedder:
    """Any OpenAI-compatible embeddings endpoint (for example the MLflow gateway)."""

    def __init__(self, model: str, base_url: str, api_key: str):
        from langchain_openai import OpenAIEmbeddings

        self.name = f"gateway:{model}"
        self._client = OpenAIEmbeddings(
            model=model, base_url=base_url, api_key=api_key, check_embedding_ctx_length=False
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)


class HashEmbedder:
    """Deterministic bag-of-words hashing. Offline, used by tests and as a last resort.

    Retrieval quality is keyword-level only; use it when no model can be loaded.
    """

    STOPWORDS = frozenset(
        "a an the and or of to in on for is are be can i my it if do does what which with "
        "within from by at as this that any all you your me we our was were will would".split()
    )
    # Hash vectors give lower absolute similarities than a trained model.
    default_min_relevance = 0.1

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}"

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        tokens = []
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in cls.STOPWORDS:
                continue
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]  # crude plural folding: phones -> phone
            tokens.append(token)
        return tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def embedder_from_env() -> Embedder:
    backend = os.getenv("POLICY_EMBEDDINGS", "local").strip().lower()
    if backend == "local":
        return LocalEmbedder()
    if backend == "gateway":
        return GatewayEmbedder(
            model=os.getenv("POLICY_EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:5001/gateway/mlflow/v1"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
        )
    if backend == "hash":
        return HashEmbedder()
    raise PolicyIndexError(f"Unknown POLICY_EMBEDDINGS backend: {backend} (use local, gateway or hash)")


# ------------------------------------------------------------------ index ----
@dataclass
class PolicyHit:
    number: int
    text: str
    relevance: float
    retailer: str
    doc_type: str
    source_id: str
    source_url: str
    retrieved_at: str
    heading: str = ""
    matched_by: tuple[str, ...] = ()  # "semantic" and/or "keyword"


@dataclass
class BuildReport:
    documents: int
    chunks: int
    by_retailer: dict[str, int]
    embedder: str


class PolicyIndex:
    def __init__(self, embedder: Embedder, persist_dir: Path | str | None = None):
        import chromadb

        self.embedder = embedder
        self.persist_dir = Path(persist_dir or os.getenv("POLICY_CHROMA_DIR") or DEFAULT_CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

    # -- build ---------------------------------------------------------------
    def build(self, documents: list[PolicyDocument], batch_size: int = 64) -> BuildReport:
        """Rebuild the index from scratch.

        The new index is written to a staging collection and only swapped in
        once it is complete, so a failure part-way leaves the old index intact.
        """
        if not documents:
            raise PolicyIndexError("No policy documents to index. Fetch or add files first.")
        chunks = [chunk for document in documents for chunk in chunk_document(document)]
        if not chunks:
            raise PolicyIndexError("Policy documents produced no text chunks.")

        self._drop(BUILD_COLLECTION_NAME)
        staging = self._client.create_collection(
            BUILD_COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": self.embedder.name},
        )
        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                staging.add(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    embeddings=self.embedder.embed([chunk.text for chunk in batch]),
                    metadatas=[chunk.metadata for chunk in batch],
                )
        except Exception:
            self._drop(BUILD_COLLECTION_NAME)
            raise

        self._drop(COLLECTION_NAME)
        staging.modify(name=COLLECTION_NAME)

        by_retailer: dict[str, int] = {}
        for chunk in chunks:
            retailer = chunk.metadata["retailer"]
            by_retailer[retailer] = by_retailer.get(retailer, 0) + 1
        return BuildReport(len(documents), len(chunks), by_retailer, self.embedder.name)

    def _drop(self, name: str) -> None:
        if name in {collection.name for collection in self._client.list_collections()}:
            self._client.delete_collection(name)

    # -- query ---------------------------------------------------------------
    def _collection(self):
        if COLLECTION_NAME not in {c.name for c in self._client.list_collections()}:
            raise PolicyIndexError(
                "The policy index has not been built. Run: python scripts/policies/build_policy_index.py"
            )
        collection = self._client.get_collection(COLLECTION_NAME, embedding_function=None)
        built_with = (collection.metadata or {}).get("embedder")
        if built_with != self.embedder.name:
            raise PolicyIndexError(
                f"The policy index was built with '{built_with}' embeddings but "
                f"'{self.embedder.name}' is configured. Rebuild the index or change POLICY_EMBEDDINGS."
            )
        return collection

    def stats(self) -> dict:
        collection = self._collection()
        rows = collection.get(include=["metadatas"])
        retailers: dict[str, set[str]] = {}
        dates: dict[str, str] = {}
        for metadata in rows["metadatas"]:
            retailers.setdefault(metadata["retailer"], set()).add(metadata["source_id"])
            dates[metadata["source_id"]] = metadata["retrieved_at"]
        return {
            "chunks": collection.count(),
            "documents": len(dates),
            "embedder": self.embedder.name,
            "retailers": {key: sorted(value) for key, value in sorted(retailers.items())},
            "oldest_retrieval": min(dates.values()) if dates else None,
            "newest_retrieval": max(dates.values()) if dates else None,
        }

    # -- keyword index --------------------------------------------------------
    def _keyword_index(self, collection) -> "KeywordIndex":
        """BM25 over every stored passage, rebuilt only when the index changes."""
        key = (getattr(collection, "id", None), collection.count())
        if getattr(self, "_keyword_cache_key", None) != key:
            rows = collection.get(include=["documents", "metadatas"])
            self._keyword_index_cache = KeywordIndex(
                rows["ids"], rows["documents"], rows["metadatas"]
            )
            self._keyword_cache_key = key
        return self._keyword_index_cache

    def search(
        self,
        question: str,
        retailers: list[str] | None = None,
        *,
        k: int = 5,
        include_regulations: bool = True,
        min_relevance: float | None = None,
        boost_terms: tuple[str, ...] = (),
        hybrid: bool | None = None,
    ) -> list[PolicyHit]:
        """Hybrid search: MiniLM (meaning) + BM25 (exact words), merged by rank.

        * Meaning-only matches must clear the relevance cut-off.
        * Passages that also contain the question's key words are kept even
          below the cut-off, so exact terms ("restocking fee", "7 days") are
          not lost when the embedding similarity is modest.
        * ``boost_terms`` (product-category words) lift passages about the
          product being bought above generic ones.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        if min_relevance is not None:
            threshold = min_relevance
        elif os.getenv("POLICY_MIN_RELEVANCE"):
            threshold = float(os.environ["POLICY_MIN_RELEVANCE"])
        else:
            threshold = getattr(self.embedder, "default_min_relevance", DEFAULT_MIN_RELEVANCE)
        if hybrid is None:
            hybrid = os.getenv("POLICY_HYBRID_SEARCH", "true").strip().lower() not in {"0", "false", "no"}
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []
        allowed = None
        where = None
        if retailers:
            allowed = set(retailers) | ({REGULATION} if include_regulations else set())
            where = {"retailer": {"$in": sorted(allowed)}}

        query_vector = self.embedder.embed([expanded_query(question)])[0]
        pool = min(max(k * 4, 20), total)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=pool,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        passages: dict[str, dict] = {}
        semantic_rank: dict[str, int] = {}
        for rank, (chunk_id, text, metadata, distance) in enumerate(zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        )):
            passages[chunk_id] = {"text": text, "metadata": metadata,
                                  "relevance": 1.0 - float(distance)}
            semantic_rank[chunk_id] = rank

        keyword_rank: dict[str, int] = {}
        if hybrid:
            ranked = self._keyword_index(collection).search(question, allowed, limit=pool)
            keyword_rank = {chunk_id: rank for rank, (chunk_id, _score) in enumerate(ranked)}
            missing = [chunk_id for chunk_id in keyword_rank if chunk_id not in passages]
            if missing:
                extra = collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
                for chunk_id, text, metadata, vector in zip(
                    extra["ids"], extra["documents"], extra["metadatas"], extra["embeddings"]
                ):
                    passages[chunk_id] = {"text": text, "metadata": metadata,
                                          "relevance": _cosine(query_vector, vector)}

        candidates = []
        for chunk_id, item in passages.items():
            in_keyword = chunk_id in keyword_rank
            if item["relevance"] < threshold and not in_keyword:
                continue
            score = 0.0
            if chunk_id in semantic_rank:
                score += 1.0 / (60 + semantic_rank[chunk_id])
            if in_keyword:
                score += 1.0 / (60 + keyword_rank[chunk_id])
            lowered = item["text"].lower()
            if any(term in lowered for term in boost_terms):
                score += 1.0 / 60  # worth one top-ranked vote
            sources = tuple(
                name for name, ranks in (("semantic", semantic_rank), ("keyword", keyword_rank))
                if chunk_id in ranks and (name == "keyword" or item["relevance"] >= threshold)
            )
            candidates.append((score, chunk_id, item, sources))
        candidates.sort(key=lambda entry: entry[0], reverse=True)

        hits: list[PolicyHit] = []
        for _score, _chunk_id, item, sources in candidates[:k]:
            metadata = item["metadata"]
            hits.append(
                PolicyHit(
                    number=len(hits) + 1,
                    text=item["text"],
                    relevance=round(item["relevance"], 4),
                    retailer=metadata["retailer"],
                    doc_type=metadata["doc_type"],
                    source_id=metadata["source_id"],
                    source_url=metadata["source_url"],
                    retrieved_at=metadata["retrieved_at"],
                    heading=metadata.get("heading", ""),
                    matched_by=sources,
                )
            )
        return hits


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


_KEYWORD_STOPWORDS = frozenset(
    "a an and any are as at be been but by can could do does for from get got has have "
    "how i if in into is it its me my of on or our so than that the their them then "
    "there this to was we were what when where which who will with would you your "
    "policy policies rule rules".split()
)


# Words retailer pages use for the same thing. A question about a "phone"
# must find a table row about "Mobiles".
SYNONYM_GROUPS = [
    {"phone", "mobile", "smartphone", "handset"},
    {"laptop", "notebook"},
    {"tablet", "ipad"},
    {"earbud", "earphone", "headphone", "headset", "tws"},
    {"tv", "television"},
    {"watch", "smartwatch", "wearable"},
    {"refund", "reimbursement"},
    {"cancel", "cancellation", "cancelled"},
    {"replace", "replacement", "exchange"},
    {"return", "returned", "returnable"},
    {"defective", "damaged", "faulty"},
]
_SYNONYMS = {word: group for group in SYNONYM_GROUPS for word in group}


def expand_terms(terms: list[str]) -> list[set[str]]:
    """Each query term becomes the set of words that count as a match for it."""
    return [set(_SYNONYMS.get(term, {term})) for term in terms]


def expanded_query(question: str) -> str:
    """Question text plus the related words it implies, for the embedding model."""
    extra = []
    tokens = set(keyword_tokens(question))
    for token in tokens:
        for word in sorted(_SYNONYMS.get(token, ())):
            if word not in tokens and word not in extra:
                extra.append(word)
    return f"{question} ({' '.join(extra)})" if extra else question


RETAILER_NAMES = {
    "amazon": "amazon",
    "flipkart": "flipkart",
    "croma": "croma",
    "reliance digital": "reliance_digital",
    "reliance": "reliance_digital",
    "vijay sales": "vijay_sales",
}


def retailers_in_question(question: str) -> list[str]:
    """Retailer keys named in a question ("... bought on Flipkart?")."""
    text = (question or "").lower()
    found: list[str] = []
    for name, key in RETAILER_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", text) and key not in found:
            found.append(key)
    return found


def keyword_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if token in _KEYWORD_STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]  # phones -> phone, days -> day
        tokens.append(token)
    return tokens


class KeywordIndex:
    """Small in-memory BM25 index over the stored policy passages."""

    def __init__(self, ids, documents, metadatas, k1: float = 1.5, b: float = 0.75):
        self.ids = list(ids)
        self.metadatas = list(metadatas)
        self.docs = [keyword_tokens(text) for text in documents]
        self.k1, self.b = k1, b
        self.avg_len = (sum(len(doc) for doc in self.docs) / len(self.docs)) if self.docs else 0.0
        self.doc_freq: dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.term_freqs = []
        for doc in self.docs:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs.append(counts)

    def search(self, question: str, allowed_retailers=None, limit: int = 20):
        terms = list(dict.fromkeys(keyword_tokens(question)))
        if not terms or not self.docs:
            return []
        groups = expand_terms(terms)
        total = len(self.docs)
        # A passage must match at least two query terms (or the only one) so
        # that a single common word like "return" does not pull in everything.
        needed = 1 if len(terms) == 1 else 2
        scored = []
        for index, counts in enumerate(self.term_freqs):
            if allowed_retailers and self.metadatas[index].get("retailer") not in allowed_retailers:
                continue
            # A query term matches when the passage contains it or a related word;
            # the best-scoring word in each group counts once.
            matched_groups = [
                [word for word in group if word in counts] for group in groups
            ]
            matched_groups = [words for words in matched_groups if words]
            if len(matched_groups) < needed:
                continue
            length = len(self.docs[index]) or 1
            score = 0.0
            for words in matched_groups:
                term = max(words, key=lambda word: counts[word])
                df = self.doc_freq[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                tf = counts[term]
                score += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                )
            scored.append((self.ids[index], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


# ---------------------------------------------------------------- answers ----
RESTRICTION_PATTERNS = {
    "replacement only (no refund)": r"replacement\s+only|only\s+(?:a\s+)?replacement",
    "non-returnable": r"non[-\s]?returnable|not\s+returnable|cannot\s+be\s+returned|no\s+returns?\b",
    "not eligible for return": r"not\s+eligible\s+for\s+(?:a\s+)?(?:return|refund)",
    "inspection / verification required": r"technician\s+visit|inspection|verification\s+by\s+(?:the\s+)?(?:brand|technician)",
    "seal / packaging must be intact": r"seal(?:ed)?\s+(?:must|should)|(?:original|intact)\s+packaging|brand\s+seal",
}


def _mentions(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term.strip())}", text) for term in terms if term.strip())


def about_other_product(hit: PolicyHit, own_terms: tuple[str, ...], other_terms: tuple[str, ...]) -> bool:
    """True when a passage is about another product type and not this one.

    Looks at the section heading and the start of the passage, where retailer
    pages name the category a rule applies to ("Laptops: ...").
    """
    if not other_terms:
        return False
    scope = f"{hit.heading} {hit.text[:300]}".lower()
    return _mentions(scope, other_terms) and not _mentions(scope, own_terms)


def detect_restrictions(hits: list[PolicyHit]) -> list[dict]:
    """Deterministic scan of retrieved passages for restrictive return terms."""
    found = []
    for label, pattern in RESTRICTION_PATTERNS.items():
        for hit in hits:
            match = re.search(pattern, hit.text, flags=re.IGNORECASE)
            if match:
                start = max(match.start() - 80, 0)
                if start:
                    start = hit.text.find(" ", start) + 1 or start  # begin on a word boundary
                found.append(
                    {
                        "restriction": label,
                        "retailer": hit.retailer,
                        "citation": hit.number,
                        "excerpt": re.sub(r"\s+", " ", hit.text[start:match.end() + 120]).strip(),
                        "source_url": hit.source_url,
                    }
                )
                break
    return found


@dataclass
class PolicyAnswer:
    question: str
    answer: str
    mode: str  # "llm" | "extractive" | "no_match"
    citations: list[PolicyHit] = field(default_factory=list)
    restrictions: list[dict] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = (
    "You are the Policy Analyst for PriceLens, an Indian electronics purchase advisor.\n"
    "Answer the user's question using ONLY the numbered policy passages provided.\n"
    "Rules:\n"
    "1. Cite every factual statement with its passage number in square brackets, e.g. [2].\n"
    "2. If the passages do not contain the answer, say exactly which part is not stated "
    "in the indexed policies. Never use outside knowledge and never guess windows, fees or dates.\n"
    "3. Name the retailer each rule belongs to. Keep the answer under 150 words."
)


def default_llm():
    from langchain_openai import ChatOpenAI

    options = {}
    # GPT-5-family and other reasoning models reject any temperature except the
    # default, so only send one when it is explicitly configured.
    if os.environ.get("POLICY_LLM_TEMPERATURE", "").strip():
        options["temperature"] = float(os.environ["POLICY_LLM_TEMPERATURE"])
    return ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:5001/gateway/mlflow/v1"),
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
        model=os.environ.get("LLM_MODEL", "gemini"),
        timeout=float(os.environ.get("POLICY_LLM_TIMEOUT", "60")),
        max_retries=0,
        **options,
    )


def _format_passages(hits: list[PolicyHit]) -> str:
    blocks = []
    for hit in hits:
        heading = f" — {hit.heading}" if hit.heading else ""
        blocks.append(
            f"[{hit.number}] retailer={hit.retailer} doc={hit.doc_type}{heading} "
            f"(retrieved {hit.retrieved_at})\n{hit.text}"
        )
    return "\n\n".join(blocks)


def validate_citations(answer: str, hits: list[PolicyHit]) -> str | None:
    """Return a problem description, or None when every citation is grounded."""
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return "the model's answer cited no passages"
    unknown = cited - {hit.number for hit in hits}
    if unknown:
        return f"the model cited passages that were not supplied: {sorted(unknown)}"
    return None


def extractive_answer(hits: list[PolicyHit], limit: int = 3, width: int = 450) -> str:
    lines = ["Relevant policy passages:"]
    for hit in hits[:limit]:
        text = re.sub(r"\s+", " ", hit.text).strip()
        if len(text) > width:
            text = text[:width].rsplit(" ", 1)[0] + " …"
        lines.append(f"- [{hit.number}] ({hit.retailer}) {text}")
    return "\n".join(lines)


class PolicyAdvisor:
    def __init__(self, index: PolicyIndex, llm="auto"):
        self.index = index
        self._llm = llm
        self._llm_error: str | None = None

    def _get_llm(self):
        if self._llm == "auto":
            try:
                self._llm = default_llm()
            except Exception as exc:  # missing package or bad config
                log.warning("Policy LLM unavailable: %s", exc)
                self._llm = None
        return self._llm

    def answer(
        self,
        question: str,
        retailers: list[str] | None = None,
        *,
        k: int = 5,
        include_regulations: bool = True,
        boost_terms: tuple[str, ...] = (),
        exclude_terms: tuple[str, ...] = (),
    ) -> PolicyAnswer:
        """``boost_terms`` describe the product being bought; ``exclude_terms``
        describe other product types, whose clauses are not flagged. When no
        retailers are given, retailers named in the question are used."""
        if not retailers:
            retailers = retailers_in_question(question) or None
        hits = self.index.search(
            question, retailers, k=k, include_regulations=include_regulations, boost_terms=boost_terms
        )
        if not hits:
            scope = ", ".join(retailers) if retailers else "any retailer"
            return PolicyAnswer(
                question,
                f"No relevant passage was found in the indexed policies for {scope}. "
                "The answer is not stated in the current corpus.",
                "no_match",
            )
        # Do not flag clauses written for a different product type (a laptop
        # "brand seal" rule is not a restriction on a phone). Generic clauses
        # that name no product type are still flagged.
        restriction_hits = [
            hit for hit in hits if not about_other_product(hit, boost_terms, exclude_terms)
        ]
        restrictions = detect_restrictions(restriction_hits)
        llm = self._get_llm()
        if llm is None:
            return PolicyAnswer(
                question, extractive_answer(hits), "extractive", hits, restrictions,
                note=(
                    f"LLM unavailable ({self._llm_error}); showing retrieved passages."
                    if self._llm_error else "LLM not configured; showing retrieved passages."
                ),
            )
        try:
            response = llm.invoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", f"Question: {question}\n\nPolicy passages:\n{_format_passages(hits)}"),
                ]
            )
            content = getattr(response, "content", response) or ""
            if isinstance(content, list):  # some providers return content blocks
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                )
            text = str(content).strip()
        except Exception as exc:
            self._llm = None  # don't wait on an unreachable LLM again this session
            self._llm_error = type(exc).__name__
            return PolicyAnswer(
                question, extractive_answer(hits), "extractive", hits, restrictions,
                note=f"LLM unavailable ({type(exc).__name__}); showing retrieved passages.",
            )
        problem = validate_citations(text, hits)
        if problem:
            return PolicyAnswer(
                question, extractive_answer(hits), "extractive", hits, restrictions,
                note=f"LLM answer rejected by the grounding check: {problem}.",
            )
        return PolicyAnswer(question, text, "llm", hits, restrictions)
