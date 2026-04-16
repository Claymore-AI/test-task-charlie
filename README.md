# Charlie AI — Vocabulary Lesson Engine

An EdTech AI service that orchestrates vocabulary mini-lessons for children aged 4–8.  
Charlie is a playful 8-year-old fox from London who teaches English words through guided conversation, using multiple specialized AI agents working together.

## Quick Start

### CLI Mode

```bash
# 1. Clone & enter the project
cd charlie-ai

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your Groq API key (free at https://console.groq.com)
cp .env.example .env
# Edit .env → set GROQ_API_KEY=gsk_...

# 5. Run the lesson
python main.py

# Or with custom words:
python main.py apple house tree
```

### API Mode (for mobile app integration)

```bash
# Start the API server
uvicorn charlie_ai.api:app --reload

# Start a lesson
curl -X POST http://localhost:8000/lesson/start \
  -H "Content-Type: application/json" \
  -d '{"words": ["cat", "dog", "bird"]}'

# Send a turn
curl -X POST http://localhost:8000/lesson/{session_id}/turn \
  -H "Content-Type: application/json" \
  -d '{"text": "cat"}'

# Check progress
curl http://localhost:8000/lesson/{session_id}/progress
```

### Run Tests

```bash
pytest tests/ -v
```
