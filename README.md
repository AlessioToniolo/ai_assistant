# AI Assistant

Chatbot with custom RAG pipeline and custom vector store. Designed to be as lightweight as possible.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlessioToniolo/ai_assistant.git
cd ai_assistant
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.sample .env
```
Then edit `.env` with your configuration:
```
ANTHROPIC_API_KEY="your-api-key-here"
OPENAI_API_KEY="your-api-key-here"
```

## Running the Application

* Start the backend server:
```bash
cd backend
python server.py
```

* Start the frontend:
```bash
cd frontend
python -m http.server 8080
```