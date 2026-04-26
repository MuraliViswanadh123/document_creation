# 📄 AutoDocs — AI Documentation Synthesizer

Generate complete 60-70 page academic reports from source code using Gemini AI.

## Features

- 📂 Upload any codebase (ZIP file)
- 🧠 Automatic project understanding via AST + Gemini
- ✍️ Generates 16-20 academic sections
- 📊 6-7 UML diagrams (PlantUML rendered to PNG)
- 📄 IEEE-formatted DOCX output
- 🚀 Deploy to Streamlit Community Cloud (free)

## Setup

### Prerequisites

- Python 3.11.x
- Gemini API key (free): https://aistudio.google.com

### Installation

```bash
# 1. Clone or download this project
cd doc-synthesizer

# 2. Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure API key
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
GEMINI_API_KEY = "your_key_here"
EOF

# 5. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

## Project Structure

```
doc-synthesizer/
├── app.py                          # Streamlit UI (entry point)
├── pipeline/
│   ├── __init__.py
│   ├── parser.py                   # Tree-sitter AST extraction
│   ├── gemini_client.py            # Gemini API wrapper
│   ├── diagram_generator.py        # PlantUML → PNG via Kroki
│   ├── docx_builder.py             # python-docx assembly
│   └── models.py                   # Pydantic schemas
├── prompts/
│   ├── abstract.j2                 # Jinja2 prompt templates
│   ├── introduction.j2
│   └── ...                         # One per section
├── templates/
│   └── ieee_template.docx          # Base IEEE-formatted template
├── requirements.txt
├── runtime.txt                     # Pins Python version for deploy
├── .gitignore
└── .streamlit/
    ├── config.toml                 # UI theme
    └── secrets.toml                # API keys (DO NOT COMMIT)
```

## Deployment to Streamlit Cloud

1. Push your project to a public GitHub repo
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → connect repo → deploy
4. In the Streamlit dashboard, add your `GEMINI_API_KEY` to Secrets
5. App is live at `https://your-app.streamlit.app`

## Architecture

```
User Code (ZIP)
       ↓
Tree-sitter AST Parser
       ↓
Gemini Project Profiler  ← Optional abstract/paper
       ↓
Document Planner
       ↓
Gemini Section Generator (×20 parallel)
       ↓
Gemini PlantUML Generator (×7)
       ↓
Kroki.io Renderer → PNG diagrams
       ↓
python-docx Assembler
       ↓
DOCX Download
```

## License

MIT
