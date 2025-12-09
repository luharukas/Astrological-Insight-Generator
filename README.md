# Astrological Insight API

A FastAPI-based service that generates personalized daily astrological insights. It determines your zodiac sign based on your birth details, produces a friendly, concise insight using a large language model (LLM), and supports translation into selected Indic languages.

## Features
- Health check endpoint to verify service availability
- Zodiac sign lookup by date
- Personalized insight generation via `/predict`, with caching of English responses
- Automatic translation of insights from English to Indic languages
- Unified LLM integration with fallback between Google Gemini and OpenAI models via `llmkit`
- Stubbed vector store for future similarity-based retrieval

## Project Structure
```
.
├── app.py                # FastAPI application entry point
├── astrology/            # Core astrology modules
│   ├── input_parser.py   # Parse and validate birth date, time, and location
│   ├── zodiac.py         # Determine zodiac sun sign
│   ├── translator.py     # Translate insights to Indic languages
│   └── vector_store.py   # In-memory stub for vector storage
├── llmkit/               # LLM integration toolkit with fallback logic
├── IndicTransToolkit/    # Embedded IndicTransToolkit library for translation support
├── requirements.txt      # Python dependencies
└── README.md             # This documentation
```

## Prerequisites
- Python 3.11 or newer
- Git
- API keys for LLM providers and translation:
  - `OPENAI_API_KEY` (OpenAI)
  - `GEMINI_API_KEY` (Google Gemini)
  - `HF_TOKEN` (Hugging Face token for IndicTrans2 model)

## Installation
1. Clone the main repository and the embedded IndicTransToolkit:
   ```bash
   git clone https://github.com/luharukas/Astrological-Insight-Generator.git
   cd Astrological-Insight-Generator
   git clone https://github.com/VarunGumma/IndicTransToolkit.git
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install --editable ./IndicTransToolkit/
   ```
4. Define environment variables (e.g., in a `.env` file at project root):
   ```ini
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   HF_TOKEN=your_huggingface_token
   ```

## Usage
### Running the API
Start the application with Uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker

Build the Docker image:

```bash
docker build -t astrological-insight-api .
```

Run the container (using a `.env` file for credentials):

```bash
docker run -d --name astro-api --env-file .env -p 8000:8000 astrological-insight-api
```

### API Endpoints
- **GET /health**  
  Check service status.  
  **Response**: `{ "status": "ok" }`

- **GET /zodiac/{date_str}**  
  Lookup zodiac sign by date (`YYYY-MM-DD`).  
  **Response**: `{ "zodiac": "Aries" }`

- **POST /predict**  
  Generate a personalized astrological insight.  
  **Request Body** (`application/json`):
  ```json
  {
    "name": "Jane",
    "birth_date": "1990-05-15",
    "birth_time": "08:30",
    "birth_place": "Mumbai, India",
    "language": "en"
  }
  ```
  **Response**:
  ```json
  {
    "zodiac": "Taurus",
    "insight": "Your day is filled with creative energy...",
    "language": "en",
    "cached": false,
    "timestamp": "2025-12-08T15:20:30.123456Z"
  }
  ```

## Supported Translation Languages
- hi (Hindi), mr (Marathi), bn (Bengali), gu (Gujarati)
- pa (Punjabi), ta (Tamil), te (Telugu), kn (Kannada)
- ml (Malayalam), or (Odia), ne (Nepali), ur (Urdu)

## Customization & Extension
- Replace the in-memory `astrology/vector_store.VectorStore` with a production-ready vector database
- Extend translation support for additional language pairs in `astrology/translator.py`

## Contributing
Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request.

## License
Please refer to `IndicTransToolkit/LICENSE` for the embedded translation toolkit license. Main project license is not specified.
