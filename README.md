# RAG Custom Assistant

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlessioToniolo/obsidianbrain.git
cd obsidianbrain
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
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```