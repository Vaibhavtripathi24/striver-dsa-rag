"""Qdrant upsert + query.

Same client as day14/day15, but with two differences that matter at this scale:
the collection name carries the embedding dim, and point IDs are derived from
a stable chunk_id so re-ingesting overwrites instead of duplicating.
"""

import atexit
import json
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ytrag.config import (
    COLLECTION,
    MAX_DISTANCE,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
    TITLE_BOOST,
    TOP_K,
    UPSERT_BATCH,
)
from ytrag.embed import get_embedder
from ytrag.models import Chunk, _NAMESPACE
from ytrag.util import with_retry

_CLIENT: QdrantClient | None = None


def _close_client() -> None:
    """Release the store before the interpreter tears down.

    Qdrant's own __del__ runs during shutdown, by which point sys.meta_path is
    gone and its close() raises ImportError. Harmless, but it prints a
    traceback after a successful command and looks like a crash.
    """
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None


atexit.register(_close_client)


def get_client() -> QdrantClient:
    """Qdrant Cloud when configured, otherwise an embedded local store.

    The local mode matters more than it looks: it means someone can clone this
    repo and have a working index with no Qdrant account, no Docker, and no
    signup — just a folder on disk. QDRANT_URL upgrades them to the hosted
    cluster whenever they want one.
    """
    global _CLIENT
    if _CLIENT is None:
        if QDRANT_URL:
            _CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            try:
                _CLIENT = QdrantClient(path=str(QDRANT_PATH))
            except RuntimeError as exc:
                # The embedded store is a single-writer file lock, so a second
                # command while `ytrag serve` is running fails with a message
                # that explains nothing. Say what actually happened.
                if "already accessed" in str(exc) or "Storage folder" in str(exc):
                    raise RuntimeError(
                        "The local index is already open in another process — most likely "
                        "`ytrag serve` is running in another terminal. Stop it (Ctrl-C) and "
                        "try again, or set QDRANT_URL to use a hosted Qdrant which allows "
                        "many readers at once."
                    ) from exc
                raise
    return _CLIENT


def collection_name() -> str:
    """e.g. 'dsa_lectures_1024'.

    Qdrant rejects vectors whose size does not match the collection, so
    stamping the dim into the name means switching embedding models creates a
    new collection instead of erroring — and lets a 1024-dim local index and a
    384-dim deploy index live side by side.
    """
    return f"{COLLECTION}_{get_embedder().dim}"


def ensure_collection() -> str:
    """Create the collection if it does not exist. Safe to call every time."""
    client = get_client()
    name = collection_name()

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=get_embedder().dim,
                distance=Distance.COSINE,
            ),
        )
        # Only meaningful on a Qdrant server — the embedded store filters
        # without an index and warns if you ask for one.
        if QDRANT_URL:
            client.create_payload_index(
                collection_name=name,
                field_name="video_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
    return name


def upsert_chunks(chunks: list[Chunk], batch_size: int = UPSERT_BATCH) -> int:
    """Embed and upsert. Idempotent: same chunk_id -> same point ID -> overwrite."""
    if not chunks:
        return 0

    name = ensure_collection()
    client = get_client()
    embedder = get_embedder()

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        points = [
            PointStruct(id=c.point_id, vector=v, payload=c.to_payload())
            for c, v in zip(batch, vectors)
        ]
        # Retried for the same reason downloads are: this is a network call in
        # the middle of a long unattended run. Upsert is idempotent, so a
        # retry after a partial success is harmless.
        with_retry(
            lambda: client.upsert(collection_name=name, points=points, wait=True),
            label=f"upsert {len(points)} points",
        )
        total += len(points)

    return total


def delete_video(video_id: str) -> None:
    """Remove every chunk for one video. Used when re-chunking with new settings."""
    client = get_client()
    name = ensure_collection()
    client.delete(
        collection_name=name,
        points_selector=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        wait=True,
    )


def indexed_video_ids() -> set[str]:
    """Which videos already have chunks in the collection.

    Lets a re-run skip the embed+upsert for work already done. Without this,
    restarting a partly-finished ingest re-embeds every cached transcript
    before reaching new material — minutes of idle GPU each time.
    """
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return set()

    found: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["video_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("video_id"):
                found.add(point.payload["video_id"])
        if offset is None:
            break
    return found


# Words that carry no topic signal — Hinglish question scaffolding, plus the
# boilerplate that appears in almost every lecture title.
_STOP = {
    "kaise", "kya", "hai", "hain", "me", "ka", "ki", "ke", "aur", "kab", "karte",
    "karna", "hota", "nikale", "solve", "kare", "chahiye", "use", "kahan", "se",
    "ko", "pehchane", "difference", "farak", "the", "a", "an", "is", "in", "what", "how",
    "do", "to", "of", "for", "with", "given", "using", "part", "find", "all", "video",
    "dsa", "patterns", "pattern", "episode", "leetcode", "interview", "questions",
    "question", "master", "best", "explained", "explain", "explanation", "problem",
    "problems", "types", "same", "code", "codes", "approach", "approaches", "brute",
    "better", "optimal", "playlist", "series", "sheet", "striver", "strivers", "a2z",
    "course", "lecture", "cpp", "c++", "java", "python", "hindi", "hinglish", "intuition",
    "technique", "method", "solution", "tell", "about", "give", "samjhao", "samjha"
}


# Pre-compiled regex patterns for zero compilation overhead during searches
RE_NUM_2 = re.compile(r"\b2\b", re.IGNORECASE)
RE_NUM_3 = re.compile(r"\b3\b", re.IGNORECASE)
RE_NUM_4 = re.compile(r"\b4\b", re.IGNORECASE)
RE_WORDS = re.compile(r"[a-z0-9]+")
RE_DELIM = re.compile(r"[\|\-\:]")

_DSA_SYNONYMS = [
    (re.compile(r"\b2\s*sum\b", re.IGNORECASE), "two sum 2 sum pair with given sum"),
    (re.compile(r"\b3\s*sum\b", re.IGNORECASE), "three sum 3 sum triplet sum"),
    (re.compile(r"\b4\s*sum\b", re.IGNORECASE), "four sum 4 sum quad sum"),
    (re.compile(r"\bkadane\b", re.IGNORECASE), "kadane algorithm maximum subarray sum"),
    (re.compile(r"\bdijkstra\b", re.IGNORECASE), "dijkstra algorithm shortest path"),
    (re.compile(r"\blru\b", re.IGNORECASE), "lru cache implement lru"),
    (re.compile(r"\blfu\b", re.IGNORECASE), "lfu cache implement lfu"),
    (re.compile(r"\blcs\b", re.IGNORECASE), "longest common subsequence lcs"),
    (re.compile(r"\blis\b", re.IGNORECASE), "longest increasing subsequence lis"),
    (re.compile(r"\bdnf\b", re.IGNORECASE), "dutch national flag sort 0s 1s 2s"),
    (re.compile(r"\bbst\b", re.IGNORECASE), "binary search tree bst"),
    (re.compile(r"\bdll\b", re.IGNORECASE), "doubly linked list dll"),
    (re.compile(r"\bmst\b", re.IGNORECASE), "minimum spanning tree prims kruskal"),
    (re.compile(r"\bkmp\b", re.IGNORECASE), "kmp algorithm string matching"),
    (re.compile(r"\bkoko\b", re.IGNORECASE), "koko eating bananas binary search"),
    (re.compile(r"\bpascal\b", re.IGNORECASE), "pascal triangle ncr"),
]


def _stem(word: str) -> str:
    """Crude plural stripping, enough to match 'hashmap' against 'HASHMAPS'."""
    for suffix in ("es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    text_norm = text.lower()
    text_norm = RE_NUM_2.sub("two 2", text_norm)
    text_norm = RE_NUM_3.sub("three 3", text_norm)
    text_norm = RE_NUM_4.sub("four 4", text_norm)
    return {
        _stem(w)
        for w in RE_WORDS.findall(text_norm)
        if w not in _STOP and (len(w) >= 2 or w.isdigit())
    }


def expand_query(query: str) -> str:
    q = query.lower().strip()
    for pattern, expansion in _DSA_SYNONYMS:
        q = pattern.sub(expansion, q)
    return q


_VIDEO_CATALOG_CACHE: dict[str, dict] | None = None


def get_video_catalog() -> dict[str, dict]:
    """Return pre-tokenized cached catalog of video_id -> {title, norm_title, terms, main_header}."""
    global _VIDEO_CATALOG_CACHE
    if _VIDEO_CATALOG_CACHE is not None:
        return _VIDEO_CATALOG_CACHE

    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return {}

    catalog: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["video_id", "video_title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            vid = payload.get("video_id")
            vtitle = payload.get("video_title")
            if vid and vtitle and vid not in catalog:
                t_lower = vtitle.lower()
                catalog[vid] = {
                    "title": vtitle,
                    "norm_title": t_lower,
                    "terms": _terms(vtitle),
                    "main_header": RE_DELIM.split(t_lower)[0].strip(),
                }
        if offset is None:
            break

    _VIDEO_CATALOG_CACHE = catalog
    return _VIDEO_CATALOG_CACHE


def get_video_titles() -> dict[str, str]:
    """Backward compatible helper returning video_id -> video_title."""
    return {vid: meta["title"] for vid, meta in get_video_catalog().items()}


def title_score(query: str, title: str) -> float:
    """Calculate universal query-title relevance score combining term coverage, exact alias matching, and phrase alignment."""
    q_raw = query.lower().strip()
    t_raw = title.lower()

    if not q_raw or not t_raw:
        return 0.0

    q_terms = _terms(query)
    t_terms = _terms(title)

    if not q_terms:
        return 0.0

    matched_terms = q_terms & t_terms
    coverage = len(matched_terms) / len(q_terms)

    exact_boost = 0.0
    if q_raw in t_raw:
        exact_boost = 0.60
    elif coverage == 1.0:
        exact_boost = 0.45
    elif coverage >= 0.66:
        exact_boost = 0.30

    if "2 sum" in q_raw and ("2 sum" in t_raw or "two sum" in t_raw):
        exact_boost = max(exact_boost, 0.60)
    elif "3 sum" in q_raw and ("3 sum" in t_raw or "three sum" in t_raw):
        exact_boost = max(exact_boost, 0.60)
    elif "4 sum" in q_raw and ("4 sum" in t_raw or "four sum" in t_raw):
        exact_boost = max(exact_boost, 0.60)
    elif "lru" in q_raw and "lru" in t_raw:
        exact_boost = max(exact_boost, 0.60)
    elif "lfu" in q_raw and "lfu" in t_raw:
        exact_boost = max(exact_boost, 0.60)

    t_main = RE_DELIM.split(t_raw)[0].strip()
    if q_raw in t_main or any(term in t_main for term in matched_terms if len(term) >= 4):
        exact_boost += 0.15

    return (coverage * 0.25) + exact_boost


def title_overlap(query: str, title: str) -> int:
    """Legacy helper for backward compatibility."""
    return len(_terms(expand_query(query)) & _terms(title))


def search(
    query: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    max_distance: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Return [(chunk, distance)] sorted best-first, already distance-filtered."""
    name = ensure_collection()
    client = get_client()
    expanded = expand_query(query)
    vector = get_embedder().embed_query(expanded)

    candidate_points = []
    seen_ids = set()

    if not video_id:
        v_results = client.query_points(
            collection_name=name,
            query=vector,
            limit=max(top_k * 30, 150),
            with_payload=True,
        ).points
        for p in v_results:
            if p.id not in seen_ids:
                candidate_points.append(p)
                seen_ids.add(p.id)

        catalog = get_video_catalog()
        matching_vids = [
            vid for vid, meta in catalog.items()
            if title_score(query, meta["title"]) >= 0.35
        ]
        if matching_vids:
            t_filter = Filter(
                should=[
                    FieldCondition(key="video_id", match=MatchValue(value=vid))
                    for vid in matching_vids
                ]
            )
            t_points = client.query_points(
                collection_name=name,
                query=vector,
                limit=100,
                with_payload=True,
                query_filter=t_filter,
            ).points
            for p in t_points:
                if p.id not in seen_ids:
                    candidate_points.append(p)
                    seen_ids.add(p.id)
    else:
        q_filter = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )
        v_results = client.query_points(
            collection_name=name,
            query=vector,
            limit=max(top_k * 20, 120),
            with_payload=True,
            query_filter=q_filter,
        ).points
        candidate_points = v_results

    cutoff = MAX_DISTANCE if max_distance is None else max_distance
    scored: list[tuple[float, float, Chunk]] = []
    for point in candidate_points:
        raw_score = getattr(point, "score", None)
        distance = (1.0 - float(raw_score)) if raw_score is not None else 0.45

        chunk = Chunk.from_payload(point.payload)
        t_boost = title_score(query, chunk.video_title)

        composite_score = distance - t_boost
        if distance <= cutoff or t_boost >= 0.35:
            scored.append((composite_score, distance, chunk))

    scored.sort(key=lambda row: row[0])
    return [(chunk, distance) for _, distance, chunk in scored[:top_k]]


def stats() -> dict:
    """Collection size plus a per-video breakdown."""
    client = get_client()
    name = collection_name()

    if not client.collection_exists(name):
        return {"collection": name, "exists": False, "chunks": 0, "videos": {}}

    info = client.get_collection(name)
    videos: dict[str, dict] = {}

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=["video_id", "video_title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            vid = payload.get("video_id", "?")
            entry = videos.setdefault(vid, {"title": payload.get("video_title", "?"), "chunks": 0})
            entry["chunks"] += 1
        if offset is None:
            break

    return {
        "collection": name,
        "exists": True,
        "chunks": info.points_count or 0,
        "dim": get_embedder().dim,
        "embed_model": get_embedder().name,
        "videos": videos,
    }


# ------------------------------------------------------------------
# Shipping a prebuilt index
# ------------------------------------------------------------------
# Embedding 2933 chunks takes a couple of minutes on a decent CPU and rather
# longer on a weak laptop. The vectors themselves are small — 2933 x 384
# float16 is about 2 MB — so committing them means someone can clone the repo
# and have a working index in seconds, without ever running the encoder over
# the corpus. They still need the model to embed their own *queries*, which is
# why the small one matters.

def export_vectors(path: Path) -> dict:
    """Dump every point's vector and payload to a compressed .npz."""
    import numpy as np

    client = get_client()
    name = collection_name()
    vectors, payloads = [], []

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, limit=512, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for point in points:
            vectors.append(point.vector)
            payloads.append(point.payload)
        if offset is None:
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=np.array(vectors, dtype=np.float16),
        payloads=np.array(json.dumps(payloads)),
        model=np.array(get_embedder().name),
        dim=np.array(get_embedder().dim),
    )
    return {"count": len(vectors), "mb": path.stat().st_size / 1024 / 1024}


def import_vectors(path: Path, batch_size: int = UPSERT_BATCH) -> dict:
    """Load a .npz built by export_vectors straight into the collection.

    Refuses to load vectors built by a different embedding model — mixing them
    would silently wreck retrieval, since a query encoded by one model is
    meaningless against another's vectors.
    """
    import numpy as np

    data = np.load(path, allow_pickle=False)
    model = str(data["model"])
    if model != get_embedder().name:
        raise RuntimeError(
            f"These vectors were built with {model}, but YTRAG_EMBED_MODEL is "
            f"{get_embedder().name}. Set YTRAG_EMBED_MODEL={model}, or run "
            f"`ytrag reindex` to rebuild with your model."
        )

    vectors = data["vectors"].astype("float32")
    payloads = json.loads(str(data["payloads"]))
    name = ensure_collection()
    client = get_client()

    for start in range(0, len(vectors), batch_size):
        chunk_payloads = payloads[start : start + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, p["chunk_id"])),
                vector=v.tolist(),
                payload=p,
            )
            for v, p in zip(vectors[start : start + batch_size], chunk_payloads)
        ]
        client.upsert(collection_name=name, points=points, wait=True)

    return {"count": len(vectors), "model": model}
