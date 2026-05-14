#!/usr/bin/env bash
# Quick start — run from repo root

set -e
echo "🎬 VidRAG startup"

# Check ollama
if ! command -v ollama &> /dev/null; then
  echo "❌ Ollama not found. Install from https://ollama.ai"
  exit 1
fi

# Pull models if missing
echo "📦 Ensuring Ollama models..."
ollama pull nomic-embed-text 2>/dev/null || true
ollama pull llama3.2 2>/dev/null || true

# Start Ollama in background if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "🚀 Starting Ollama..."
  ollama serve &
  sleep 3
fi

# Install Python deps
echo "🐍 Installing Python deps..."
pip install -r backend/requirements.txt -q

# Start backend
echo "⚡ Starting FastAPI on :8000"
cd backend
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start frontend
echo "🌐 Starting Vite dev server on :5173"
cd frontend
npm install -q 2>/dev/null
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Backend running (PID $BACKEND_PID) at http://localhost:8000"
echo "✅ Frontend running (PID $FRONTEND_PID) at http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"
wait $BACKEND_PID
