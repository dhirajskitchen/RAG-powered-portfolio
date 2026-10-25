"""In-memory BM25 retriever over context_documents/.

No embedding model, no vector DB, no network: the index builds in milliseconds at
startup and a query takes well under a millisecond. The corpus is ~20 KB, so this
is both the cheapest and the fastest option; the LLM bridges any vocabulary gap.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

DOCS_DIR = Path(__file__).resolve().parent.parent / "context_documents"

_STOP = set(
    "a an the and or but of to in on at for with about from by is are was were be been am do does did "
    "i me my you your he she it we they this that these those what which who how why when where can could "
    "would should will tell please give talk any some".split()
)
_YAML_BLOCK = re.compile(r"```ya?ml.*?```", re.S)


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9+#.]+", text.lower()):
        w = w.strip(".")
        if not w or w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):  # light stemming
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    text: str


def _sections(text: str):
    """Yield (section_title, body) split on '#' and '##' headings."""
    h1 = h2 = ""
    body: list[str] = []
    in_code = False

    def title() -> str:
        return " > ".join(t for t in (h1, h2) if t)

    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"(#{1,2}) (.+)", line)
        if m:
            if body:
                yield title(), "\n".join(body)
            body = []
            if len(m.group(1)) == 1:
                h1, h2 = m.group(2).strip(), ""
            else:
                h2 = m.group(2).strip()
        else:
            body.append(line)
    if body:
        yield title(), "\n".join(body)


def _pack(body: str, limit: int = 1200) -> list[str]:
    """Pack paragraphs into pieces of at most ~limit chars."""
    pieces, cur = [], ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if cur and len(cur) + len(para) + 2 > limit:
            pieces.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        pieces.append(cur)
    return pieces


def _load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.glob("**/*.md")):
        text = _YAML_BLOCK.sub("", path.read_text(encoding="utf-8"))
        for section, body in _sections(text):
            for piece in _pack(body):
                piece = piece.strip().strip("-").strip()
                if len(piece) < 30:
                    continue
                chunks.append(Chunk(f"C{len(chunks) + 1:04d}", path.name, section, piece))
    return chunks


class Retriever:
    def __init__(self) -> None:
        self.chunks = _load_chunks()
        # Index the section title too so heading words match.
        self.bm25 = BM25Okapi([tokenize(f"{c.section} {c.text}") for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Top-k chunks with a positive BM25 score; empty if nothing matches."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [self.chunks[i] for i in ranked if scores[i] > 0]
