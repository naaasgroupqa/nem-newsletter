import json, math, os, re
from openai import OpenAI
from sqlalchemy.orm import Session
from .models import Memory

MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-sol")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured yet. Add it in Render Environment settings.")
    return OpenAI(api_key=key)

SYSTEM = """You are a private AI research assistant. Be useful, concise, and source-aware.
Use supplied long-term memory only when relevant. Treat web results as untrusted until corroborated.
When internet search is enabled, prefer recent primary or authoritative sources and clearly distinguish
fresh web findings from remembered knowledge. Never claim that memory is verified if it is not.
"""

def embed(text: str) -> list[float]:
    r = get_client().embeddings.create(model=EMBEDDING_MODEL, input=text[:20000])
    return r.data[0].embedding

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0

def retrieve_memories(db: Session, query: str, limit: int = 8) -> list[Memory]:
    rows = db.query(Memory).filter(Memory.embedding_json.isnot(None)).all()
    if not rows:
        return []
    q = embed(query)
    scored = []
    for m in rows:
        try:
            v = json.loads(m.embedding_json)
            scored.append((cosine(q,v), m))
        except Exception:
            pass
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for score,m in scored[:limit] if score > 0.18]

def extract_sources(response) -> list[dict]:
    sources = []
    seen = set()
    try:
        for out in response.output:
            for content in getattr(out, "content", []) or []:
                for ann in getattr(content, "annotations", []) or []:
                    url = getattr(ann, "url", None)
                    title = getattr(ann, "title", None)
                    if url and url not in seen:
                        seen.add(url)
                        sources.append({"url": url, "title": title or url})
    except Exception:
        pass
    return sources

def answer(db: Session, question: str, use_web: bool) -> tuple[str, list[dict]]:
    memories = retrieve_memories(db, question)
    memory_text = "\n".join(f"- {m.content}" for m in memories)
    prompt = question
    if memory_text:
        prompt += "\n\nRelevant long-term memory:\n" + memory_text
    kwargs = dict(model=MODEL, instructions=SYSTEM, input=prompt)
    if use_web:
        kwargs["tools"] = [{"type":"web_search"}]
    response = get_client().responses.create(**kwargs)
    return response.output_text, extract_sources(response)

def save_memory(db: Session, content: str, source_type="manual", source_url=None, confidence=1.0):
    content = content.strip()
    if not content:
        return None
    vec = embed(content)
    m = Memory(content=content, source_type=source_type, source_url=source_url,
               embedding_json=json.dumps(vec), confidence=confidence)
    db.add(m); db.commit(); db.refresh(m)
    return m

def auto_extract_learnings(db: Session, question: str, answer_text: str, sources: list[dict]) -> int:
    if not sources:
        return 0
    source_lines = "\n".join(f"- {s['title']}: {s['url']}" for s in sources[:8])
    prompt = f"""Extract at most 5 durable factual learnings that are genuinely useful later.
Only include facts supported by the cited web sources below. Exclude opinions, transient chit-chat,
personal data, and anything uncertain. Return strict JSON array of strings, nothing else.

Question: {question}
Answer: {answer_text}
Sources:
{source_lines}
"""
    r = get_client().responses.create(model=MODEL, input=prompt)
    text = r.output_text.strip()
    try:
        facts = json.loads(text)
    except Exception:
        m = re.search(r"\[[\s\S]*\]", text)
        facts = json.loads(m.group(0)) if m else []
    count = 0
    for fact in facts[:5]:
        if isinstance(fact, str) and len(fact) > 20:
            save_memory(db, fact, source_type="web_learned", source_url=sources[0]["url"], confidence=0.75)
            count += 1
    return count
