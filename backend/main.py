"""
RAG Chatbot Backend — FastAPI + LangChain + ChromaDB + Ollama
============================================================
Run: uvicorn main:app --reload --port 8000
"""

# SSL fix for corporate proxies — must be FIRST
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import asyncio
import json
import logging
import os
import re
import time
from typing import AsyncGenerator, Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vidrag")

import chromadb
import httpx
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
# Document import removed — unused
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi  # used in fetch_transcript
import yt_dlp

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_BASE   = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
CHAT_MODEL    = "llama3.2"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 60
TOP_K         = 6
VERSION       = "1.0.0"

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Video Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chroma ───────────────────────────────────────────────────────────────────
log.info("Initializing ChromaDB (in-memory)...")
chroma_client = chromadb.Client()

log.info(f"Setting up Ollama embedding function (model={EMBED_MODEL})...")
embed_fn = OllamaEmbeddingFunction(
    url=f"{OLLAMA_BASE}/api/embeddings",
    model_name=EMBED_MODEL,
)

collection = chroma_client.get_or_create_collection(
    name="videos",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},
)
log.info("ChromaDB collection ready.")

# ── In-memory stores ──────────────────────────────────────────────────────────
video_store: dict[str, dict] = {}
conversation_history: list[dict] = []

# ── Models ───────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    url_a: str
    url_b: str

class ChatRequest(BaseModel):
    message: str
    reset: bool = False

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    patterns = [r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def fetch_transcript(video_id: str) -> str:
    log.info(f"  [transcript] Fetching for video_id={video_id} ...")
    t0 = time.time()
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        transcript = None
        # Try English first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except Exception:
            # Fall back to any available transcript
            for t in transcript_list:
                transcript = t
                break

        if not transcript:
            log.warning(f"  [transcript] No transcript available for {video_id}")
            return ""

        # Translate to English if needed and possible
        if transcript.language_code != 'en' and transcript.is_translatable:
            try:
                transcript = transcript.translate('en')
            except Exception:
                log.warning(f"  [transcript] Translation failed, using {transcript.language}")

        # fetch() returns FetchedTranscript with .snippets list
        # each snippet is a FetchedTranscriptSnippet dataclass with .text, .start, .duration
        fetched = transcript.fetch()
        snippets = fetched.snippets
        text = " ".join(s.text for s in snippets)
        log.info(f"  [transcript] OK — {len(snippets)} segments, {len(text):,} chars in {time.time()-t0:.1f}s")
        return text
    except Exception as e:
        log.error(f"  [transcript] FAILED: {e}")
        return ""

def fetch_metadata(url: str) -> dict:
    log.info(f"  [metadata] yt-dlp fetching: {url}")
    t0 = time.time()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "nocheckcertificate": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        log.info(f"  [metadata] OK — \"{info.get('title','?')}\" in {time.time()-t0:.1f}s")
    except Exception as e:
        log.error(f"  [metadata] FAILED: {e}")
        raise

    likes    = info.get("like_count")    or 0
    views    = info.get("view_count")    or 1
    comments = info.get("comment_count") or 0
    duration = info.get("duration")      or 0
    engagement = round((likes + comments) / views * 100, 4)

    log.info(f"  [metadata] views={views:,} likes={likes:,} comments={comments:,} eng={engagement}%")

    return {
        "title":          info.get("title", "Unknown"),
        "creator":        info.get("uploader", "Unknown"),
        "channel_id":     info.get("channel_id", ""),
        "follower_count": info.get("channel_follower_count") or info.get("uploader_id", "N/A"),
        "views":          views,
        "likes":          likes,
        "comments":       comments,
        "upload_date":    _fmt_date(info.get("upload_date", "")),
        "duration_sec":   duration,
        "duration_str":   f"{duration // 60}m {duration % 60}s",
        "hashtags":       info.get("tags", [])[:10],
        "thumbnail":      info.get("thumbnail", ""),
        "engagement_rate": engagement,
        "description":    (info.get("description") or "")[:500],
    }


def _fmt_date(raw: str) -> str:
    """Convert YYYYMMDD → YYYY-MM-DD, return as-is if unparseable."""
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw or "unknown"


def chunk_and_embed(video_id: str, transcript: str, metadata: dict):
    if not transcript or not transcript.strip():
        log.info("  [chunk] No transcript to chunk.")
        return 0

    log.info(f"  [chunk] Splitting transcript ({len(transcript):,} chars)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[". ", "? ", "! ", "\n", " "],
    )
    docs = splitter.create_documents(
        [transcript],
        metadatas=[{"video_id": video_id, "title": metadata["title"]}],
    )
    log.info(f"  [chunk] {len(docs)} chunks created")

    ids   = [f"{video_id}_chunk_{i}" for i in range(len(docs))]
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]

    # ── Save chunks to file for debugging ────────────────────────────────
    debug_path = os.path.join(os.path.dirname(__file__), "chunks_debug.json")
    existing = []
    if os.path.exists(debug_path):
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    # Remove old chunks for this video_id (in case of re-ingest)
    existing = [c for c in existing if c.get("video_id") != video_id]
    for i, doc in enumerate(docs):
        existing.append({
            "chunk_id": ids[i],
            "video_id": video_id,
            "title": metadata["title"],
            "chunk_index": i,
            "char_count": len(doc.page_content),
            "text": doc.page_content,
        })
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    log.info(f"  [debug] Chunks saved to {debug_path}")

    log.info(f"  [embed] Sending to Ollama nomic-embed-text ({len(docs)} chunks)...")
    t0 = time.time()
    collection.upsert(ids=ids, documents=texts, metadatas=metas)
    log.info(f"  [embed] OK — stored in ChromaDB in {time.time()-t0:.1f}s")
    return len(docs)


def retrieve(query: str, video_filter: Optional[str] = None) -> list[dict]:
    where = {"video_id": video_filter} if video_filter else None
    log.info(f"  [retrieve] \"{query[:60]}\" filter={video_filter}")
    # Cap n_results to actual count — ChromaDB throws if you ask for more than exist
    total = collection.count()
    if total == 0:
        log.warning("  [retrieve] Collection is empty — no chunks to query")
        return []
    n = min(TOP_K, total)
    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "video_id": meta.get("video_id", "?"),
            "title":    meta.get("title", "?"),
            "score":    round(1 - dist, 3),
        })
    log.info(f"  [retrieve] {len(chunks)} chunks: " +
             ", ".join(f"vid_{c['video_id']}={c['score']}" for c in chunks))
    return chunks


def build_system_prompt() -> str:
    parts = ["You are a video content analyst with access to transcript chunks and metadata.\n"]
    for vid_id, info in video_store.items():
        m = info["metadata"]
        parts.append(
            f"Video {vid_id.upper()} — \"{m['title']}\"\n"
            f"  Creator: {m['creator']} | Followers: {m['follower_count']}\n"
            f"  Views: {m['views']:,} | Likes: {m['likes']:,} | Comments: {m['comments']:,}\n"
            f"  Engagement Rate: {m['engagement_rate']}%\n"
            f"  Duration: {m['duration_str']} | Uploaded: {m['upload_date']}\n"
            f"  Hashtags: {', '.join(m['hashtags'][:5]) or 'none'}\n"
        )
    parts.append(
        "\nRules:\n"
        "- Always cite which video chunk you're drawing from: [Video A] or [Video B]\n"
        "- Be concise but analytical. Back claims with numbers.\n"
        "- When comparing, reference specific transcript excerpts.\n"
        "- If asked about the hook, focus on the first chunks of each video.\n"
    )
    return "\n".join(parts)


llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE, timeout=120)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "videos_loaded": list(video_store.keys()),
        "chunk_count": collection.count(),
    }


def process_video(label: str, url: str):
    log.info(f"\n── Video {label.upper()} ──────────────────────────────")
    log.info("  [id] Extracting video ID...")
    vid_id = extract_video_id(url)
    log.info(f"  [id] video_id = {vid_id}")

    metadata   = fetch_metadata(url)
    transcript = fetch_transcript(vid_id)
    n_chunks   = chunk_and_embed(label, transcript, metadata)
    
    return {
        "label": label,
        "url": url,
        "video_id": vid_id,
        "metadata": metadata,
        "transcript": transcript,
        "n_chunks": n_chunks
    }

@app.post("/ingest")
async def ingest(req: IngestRequest):
    log.info("=" * 55)
    log.info("INGEST REQUEST")
    log.info(f"  URL A: {req.url_a}")
    log.info(f"  URL B: {req.url_b}")
    log.info("=" * 55)

    results = {}
    errors  = {}

    loop = asyncio.get_running_loop()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_a = loop.run_in_executor(executor, process_video, "a", req.url_a)
        f_b = loop.run_in_executor(executor, process_video, "b", req.url_b)
        res_a, res_b = await asyncio.gather(f_a, f_b, return_exceptions=True)

    for res, label in [(res_a, "a"), (res_b, "b")]:
        if isinstance(res, Exception):
            log.error(f"  VIDEO {label.upper()} FAILED: {res}")
            errors[label] = str(res)
        else:
            video_store[label] = {
                "url": res["url"], "video_id": res["video_id"],
                "metadata": res["metadata"], "transcript": res["transcript"],
                "n_chunks": res["n_chunks"],
            }
            results[label] = {
                "title":           res["metadata"]["title"],
                "creator":         res["metadata"]["creator"],
                "views":           res["metadata"]["views"],
                "likes":           res["metadata"]["likes"],
                "comments":        res["metadata"]["comments"],
                "engagement_rate": res["metadata"]["engagement_rate"],
                "duration":        res["metadata"]["duration_str"],
                "thumbnail":       res["metadata"]["thumbnail"],
                "n_chunks":        res["n_chunks"],
            }
            log.info(f"  VIDEO {label.upper()} COMPLETE: \"{res['metadata']['title']}\" | {res['n_chunks']} chunks")

    log.info("\n" + "=" * 55)
    if errors:
        log.error(f"Ingest finished with errors: {errors}")
        raise HTTPException(status_code=422, detail={"errors": errors, "partial": results})

    log.info("INGEST COMPLETE — both videos ready")
    conversation_history.clear()
    return {"status": "ok", "videos": results}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not video_store:
        raise HTTPException(status_code=400, detail="No videos ingested yet.")

    if req.reset:
        conversation_history.clear()
        log.info("[chat] History cleared")
        return {"status": "reset"}

    query = req.message
    log.info(f"\n[chat] User: \"{query[:80]}\"")

    chunks = retrieve(query)

    context_lines = [
        f"[Video {c['video_id'].upper()} | score={c['score']}]\n{c['text']}"
        for c in chunks
    ]
    rag_context = "\n\n---\n\n".join(context_lines)

    augmented_prompt = (
        f"Relevant transcript excerpts:\n{rag_context}\n\n"
        f"Question: {query}\n\n"
        f"Answer based on the above excerpts and the metadata in your system prompt."
    )

    system  = build_system_prompt()
    sources = [
        {"video_id": c["video_id"].upper(), "excerpt": c["text"][:120] + "...", "score": c["score"]}
        for c in chunks[:3]
    ]

    history_msgs = []
    for msg in conversation_history:
        if msg["role"] == "user":
            history_msgs.append(HumanMessage(content=msg["content"]))
        else:
            history_msgs.append(AIMessage(content=msg["content"]))

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = prompt_template | llm | StrOutputParser()

    async def generate() -> AsyncGenerator[bytes, None]:
        full_response = []
        yield (json.dumps({"type": "sources", "data": sources}) + "\n").encode()

        try:
            log.info(f"  [langchain] Streaming from {CHAT_MODEL}...")
            async for chunk in chain.astream({
                "history": history_msgs,
                "question": augmented_prompt
            }):
                full_response.append(chunk)
                yield (json.dumps({"type": "token", "data": chunk}) + "\n").encode()

            # Only save to history if we got a complete response
            assembled = "".join(full_response)
            if assembled.strip():
                conversation_history.append({"role": "user",      "content": query})
                conversation_history.append({"role": "assistant", "content": assembled})

                if len(conversation_history) > 20:
                    del conversation_history[:2]

            log.info(f"[chat] Done — {len(assembled)} chars, history={len(conversation_history)//2} turns")
        except Exception as e:
            log.error(f"[chat] Streaming error: {e}")
            yield (json.dumps({"type": "token", "data": f"\n\n⚠️ Error: {e}"}) + "\n").encode()

        yield (json.dumps({"type": "done"}) + "\n").encode()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/videos")
def get_videos():
    out = {}
    for vid_id, info in video_store.items():
        out[vid_id] = {**info["metadata"], "n_chunks": info["n_chunks"]}
    return out


@app.delete("/reset")
def reset():
    video_store.clear()
    conversation_history.clear()
    collection.delete(where={"video_id": {"$in": ["a", "b"]}})
    log.info("[reset] Cleared all data")
    return {"status": "reset"}