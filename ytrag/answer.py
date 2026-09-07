"""Retrieve -> grounded answer + citations.

The important part of this module is what it does when retrieval comes back
empty: it returns the refusal without calling the LLM at all.

The model knows DSA perfectly well. If it is handed junk context it will
happily answer from its own training and attach your timestamps to it — the
student clicks the link and you are talking about something else entirely.
That is strictly worse than saying "cover nahi hua".
"""

import re

from groq import Groq

from ytrag.config import (
    CONFIDENT_DISTANCE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_BACKEND,
    LLM_MODEL,
    MAX_DISTANCE,
    REFUSAL,
    TOP_K,
)
from ytrag.index import _terms, search, title_overlap
from ytrag.models import Chunk

_CLIENT: Groq | None = None

SYSTEM_PROMPT = f"""You are an expert DSA mentor explaining concepts directly from Striver's (take U forward) A2Z DSA course lectures.

Format your response in a crystal-clear, structured, easy-to-understand manner (matching the language of the student's question - Hinglish or English):

### 📌 1. Problem Explanation & Example
Explain what the question is asking in simple, clear terms. Provide a simple input & expected output example so a beginner immediately understands the problem.

### 💡 2. Intuition & Core Logic
Explain the step-by-step thinking process. Why does the optimal approach work and how do we arrive at it?

### ⚙️ 3. Step-by-Step Approaches
- **Brute Force**: Explain the naive approach and why it takes more time/space.
- **Better Approach**: Explain intermediate optimizations if any.
- **Optimal Approach**: Explain the best strategy (e.g. 2 Pointers, Hash Map, Binary Search, DP, Sliding Window).

### 💻 4. Code Implementation
Provide complete, clean, well-commented code snippet for the optimal solution in the requested language (default to C++). Add comments explaining crucial logic lines.

### ⏱️ 5. Complexity Analysis
Provide a Markdown Table for complexity:
| Approach | Time Complexity | Auxiliary Space |
|---|---|---|
| Brute Force | O(...) | O(...) |
| Optimal | O(...) | O(...) |

### 🎯 6. Key Takeaway & Striver's Tip
A 1-2 sentence quick summary of the core pattern to remember for interviews.

Rules:
- Make explanations simple, thorough, warm, and easy to grasp.
- Ground your response strictly in the provided lecture excerpts. Cite excerpts using [1], [2] inline.
- If the topic is not covered in the excerpts, say exactly: "{REFUSAL}"
"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the repo-root .env.")
        _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT


def _chat(system: str, user: str) -> str:
    """One completion, from whichever backend is configured.

    Kept deliberately small: the explanation is a garnish on top of retrieval,
    so swapping providers should never be more than this function.
    """
    backend = LLM_BACKEND.lower()

    if backend == "none":
        raise RuntimeError("Explanations are disabled (YTRAG_LLM_BACKEND=none).")

    if backend == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=LLM_MODEL or GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.2
            ),
        )
        return (response.text or "").strip()

    response = get_client().chat.completions.create(
        model=LLM_MODEL or GROQ_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f'[{i}] "{chunk.video_title}" @ {chunk.timestamp}\n{chunk.text}')
    return "\n\n".join(blocks)


def _citation(chunk: Chunk, distance: float) -> dict:
    return {
        "title": chunk.video_title,
        "timestamp": chunk.timestamp,
        "url": chunk.url,
        "start_sec": chunk.link_sec,
        "video_id": chunk.video_id,
        "distance": round(distance, 4),
    }


def _renumber(text: str, hits: list[tuple[Chunk, float]]) -> tuple[str, list[dict]]:
    """Keep only the citations the model actually used, and renumber them 1..N.

    Without this the student sees six links under an answer that only used
    one, and stops trusting any of them.
    """
    order: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= len(hits) and idx not in order:
            order.append(idx)

    if not order:
        return text, []

    remap = {old: new for new, old in enumerate(order, start=1)}
    rewritten = _CITATION_RE.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "",
        text,
    )
    citations = [_citation(*hits[old - 1]) for old in order]
    return rewritten, citations


_OOD_TERMS = {
    "langchain", "langgraph", "lang graph", "autogen", "crewai", "llamaindex", "ollama", "gpt",
    "react", "angular", "vue", "node", "nodejs", "express", "django", "flask",
    "fastapi", "spring", "springboot", "flutter", "reactnative", "android", "ios",
    "cooking", "recipe", "cricket", "football", "movie", "song", "lyrics"
}


def _is_confident(question: str, hits: list[tuple[Chunk, float]]) -> bool:
    """Is the top result trustworthy enough to present without a caveat?"""
    if not hits:
        return False

    q_norm = question.lower()
    for ood in _OOD_TERMS:
        if ood in q_norm:
            return False

    chunk, distance = hits[0]
    q_terms = _terms(question)

    if not q_terms:
        return distance <= CONFIDENT_DISTANCE

    t_terms = _terms(chunk.video_title)
    overlap = len(q_terms & t_terms)
    coverage = overlap / len(q_terms)

    if coverage >= 0.50 or overlap >= 1 or distance <= CONFIDENT_DISTANCE:
        return True

    return False


def answer(
    question: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    max_distance: float | None = None,
    code_lang: str = "C++",
) -> dict:
    """-> {"answer", "citations", "grounded", "retrieved"}"""
    question = question.strip()
    if not question:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    hits = search(question, top_k=top_k, video_id=video_id, max_distance=max_distance)

    # Guard zero: if the top hit is not confident (e.g. out of domain query), refuse immediately
    if not hits or not _is_confident(question, hits):
        refusal_msg = f"Yeh topic ('{question}') Striver ke A2Z DSA course me cover nahi hua hai. Striver's A2Z DSA course me Data Structures & Algorithms (Arrays, Binary Search, Trees, Graphs, DP, etc.) covered hai."
        return {"answer": refusal_msg, "citations": [], "grounded": False, "retrieved": 0}

    chunks = [chunk for chunk, _ in hits]
    user_prompt = f"EXCERPTS\n{build_context(chunks)}\n\nQUESTION: {question}\n\n[USER PREFERENCE: Please write the code solution in {code_lang}]"

    text = _chat(SYSTEM_PROMPT, user_prompt)

    if REFUSAL.lower() in text.lower():
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": len(hits)}

    text, citations = _renumber(text, hits)

    return {
        "answer": text,
        "citations": citations,
        "grounded": bool(citations),
        "retrieved": len(hits),
    }


def retrieve_only(question: str, top_k: int = TOP_K, filtered: bool = False) -> list[tuple[Chunk, float]]:
    """Retrieval without the LLM — used by evaluate.py and `ytrag search`."""
    return search(question, top_k=top_k, max_distance=None if filtered else 2.0)


def search_only(question: str, top_k: int = TOP_K, video_id: str | None = None) -> dict:
    """Retrieval with no LLM at all — the timestamps, ranked.

    If the query is out-of-domain (e.g. 'langchain', 'react', 'lang graph') and the best match
    is not confident, return empty results so no misleading video is shown.
    """
    question = question.strip()
    if not question:
        return {"results": [], "confident": False, "query": question}

    hits = search(question, top_k=top_k, video_id=video_id)
    confident = _is_confident(question, hits)

    if not confident:
        return {
            "query": question,
            "confident": False,
            "results": [],
        }

    return {
        "query": question,
        "confident": True,
        "results": [
            {
                "title": chunk.video_title,
                "timestamp": chunk.timestamp,
                "url": chunk.url,
                "start_sec": chunk.link_sec,
                "end_sec": chunk.end_sec,
                "video_id": chunk.video_id,
                "distance": round(distance, 4),
                "preview": chunk.text.split(chr(10) + chr(10), 1)[-1][:240].strip(),
            }
            for chunk, distance in hits
        ],
    }
