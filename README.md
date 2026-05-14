# VidRAG

Compare two YouTube videos side-by-side using RAG. Paste URLs, it pulls transcripts + metadata, chunks and embeds them into ChromaDB, then you chat with an LLM that actually cites which video it's pulling from.

Everything runs locally — Ollama for both embeddings and chat, zero API keys.

## Key Features

- **Automated Comparison**: The app calculates engagement rates on the fly and highlights the "winner" so you get an immediate visual on which video performed better before you even start chatting.
- **RAG Citations**: To keep the AI grounded, every response includes source chips. You can hover over them to see the exact transcript excerpt and the similarity score the vector DB returned.
- **Performance Telemetry**: Since RAG can be slow on local hardware, I added real-time tracking for Time-to-First-Token (TTFT), DB latency, and tokens-per-second. It's tucked into a hover tooltip to keep the UI clean.
- **Zero-Config Local Stack**: Everything runs on your machine using in-memory ChromaDB and Ollama. No API keys, no cloud bills, and no data leaves your device.


## How it works

1. You paste two YouTube URLs
2. Backend fetches metadata via `yt-dlp` (views, likes, comments, creator, followers, hashtags, duration, upload date)
3. Transcripts come from `youtube-transcript-api` — auto-translated to English if needed
4. Engagement rate is calculated: `(likes + comments) / views × 100`
5. Transcripts are chunked (400 chars, 60 overlap) and embedded with `nomic-embed-text` into ChromaDB
6. Every chunk is tagged with `video_id` (A or B) so retrieval knows which video it came from
7. When you ask a question, top-6 chunks are retrieved, injected into the prompt alongside metadata, and streamed back with source citations

## Stack

- **Backend:** FastAPI (Python)
- **Frontend:** React + Vite
- **LLM:** Ollama — `llama3.2` (3B params, fast inference, fits in 4GB RAM)
- **Embeddings:** `nomic-embed-text` via Ollama (768-dim, runs locally)
- **Vector DB:** ChromaDB (in-memory)
- **Orchestration:** LangChain (ChatOllama, RecursiveCharacterTextSplitter, ChatPromptTemplate, streaming)
- **Transcripts:** `youtube-transcript-api` + `yt-dlp` for metadata

## Why these choices

**ChromaDB in-memory** — Zero setup friction. No Docker containers, no cloud accounts. The tradeoff is data doesn't survive restarts, but for a demo/dev workflow that's fine. Swapping to persistent Chroma or Qdrant is a one-line change.

**llama3.2 (3B)** — Chose this because it fits comfortably in ~4GB RAM while still giving decent analytical responses. Larger models like `gemma2:9b` or `llama3.1:8b` give better reasoning but need 8GB+ free memory. On machines with more VRAM, just change `CHAT_MODEL` in `main.py`.

**400 char chunks / 60 overlap** — Transcript text is conversational, not structured. Smaller chunks give more precise retrieval for questions like "compare the hooks" where you need the first few seconds specifically. Bigger chunks would give more context per hit but make retrieval noisier.

**nomic-embed-text** — Top-tier on MTEB benchmarks, 768 dimensions, runs entirely through Ollama. No OpenAI key needed.



## What you can ask

- "Why did Video A get more engagement than Video B?"
- "Compare the hooks in the first 5 seconds"
- "What's the engagement rate of each video?"
- "Who's the creator of Video B and what's their follower count?"
- "Suggest improvements for B based on what worked in A"

Chat maintains memory across turns so you can have follow-up conversations. 

## Challenges I ran into

**Transcript availability is hit-or-miss.** Some YouTube videos just don't have captions. Auto-generated ones exist for most English content, but foreign language videos without auto-translate are a dead end. I added fallback logic to try English first, then any available language, then translate — but it's not bulletproof. With better infra, you'd pipe the audio through Whisper as a fallback.

**yt-dlp metadata extraction is slow.** Each video takes 3-8 seconds for metadata alone. I parallelized both videos with `ThreadPoolExecutor` which helps, but at scale you'd want a job queue (Celery/Redis) and a metadata cache so you're not re-fetching the same video.

**Memory is tight on small machines.** Tried running `gemma2:9b` (9B params) for better reasoning quality — it needs 7.7GB and didn't fit. Stuck with `llama3.2` (3B) at ~2GB. The quality tradeoff is noticeable on complex comparisons, but it works. With a machine that has 16GB+ RAM or a dedicated GPU, the jump to 7B-9B models makes a real difference.

**ChromaDB in-memory won't scale.** Works great for 2 videos and ~50-100 chunks, but if you imagine 10,000 users each loading video pairs, you'd blow through memory. The move would be Qdrant or pgvector with persistent storage, plus per-user namespacing.

**Ollama is single-threaded on inference.** One user chatting = fine. Ten concurrent users = they queue up behind each other. At scale you'd need vLLM or TGI for batched inference, or just use a hosted API.

**Conversation memory is global.** Right now there's one shared `conversation_history` list. Two browser tabs = cross-talk. Fix: session tokens or user IDs with Redis-backed per-session storage.

**Comment counts are often missing.** YouTube doesn't always return `comment_count` in the API response (especially for shorts). I default to 0 but it makes the engagement rate slightly undercount. No great workaround without the full YouTube Data API.

