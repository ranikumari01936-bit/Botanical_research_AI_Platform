# Botanical Research AI Platform

A Streamlit-based application for botanical literature search and AI-assisted research, with an optional local flower-image classifier.

## Overview

This project helps users:

- Search botanical names and questions across Europe PMC and PubMed
- Resolve simple plant names using GBIF taxonomic data
- Review bibliographic metadata, abstracts, and source links
- Generate literature-review summaries with cited references
- Download eligible original PDFs when a free source PDF is available
- Classify common flower images using a local ResNet18 model

## Key Features

- Search and rank botanical literature with configurable result limits
- Deduplicate and cache search results for faster repeat queries
- Resolve ambiguous common names with GBIF scientific-name matching
- Inspect source metadata, abstracts, identifiers, and full-text links
- Generate AI-assisted review content grounded in retrieved records
- Export review content as Markdown or PDF
- Optional five-class flower classification: Daisy, Dandelion, Rose, Sunflower, Tulip

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`
- OpenAI API key for AI review and extraction features
- Internet access for literature, taxonomy, and model requests

## Setup

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `OPENAI_API_KEY` in `.env`. The app uses the OpenAI SDK default endpoint, and `BOTANICAL_AI_MODEL` can be optionally overridden.

## Run the App

```powershell
python -m streamlit run .\app.py
```

Then open the local URL shown by Streamlit, typically:

`http://localhost:8501`

## Workspaces

### Botanical Research

Use the research workspace to search by plant name, genus, species, or broader botanical question. The app supports result review, source verification, and AI-assisted literature synthesis.

### Flower Identification

Upload a flower image to classify it among the five supported classes. Results are intended for research assistance only and should not be treated as a general botanical identification system.

## Data Sources and Limitations

- Literature searches are based on Europe PMC and PubMed
- GBIF is used for short-name taxonomic matching and synonym resolution
- AI-generated summaries are research aids and must be checked against source material
- Free PDF downloads are only permitted from approved public sources
- The flower classifier is limited to five classes and does not identify arbitrary plants

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v -p "test_*.py"
.\.venv\Scripts\python.exe -m compileall -q .
```

## Project Structure

- `app.py` — Streamlit app and user workflow
- `literature_search.py` — literature retrieval, GBIF resolution, and PDF validation
- `workflow.py` — research and review workflow logic
- `extraction_agent.py` and `protocol_generator.py` — AI extraction and synthesis
- `report_pdf.py` — PDF generation for review outputs
- `flower_classifier.py` — five-class flower inference pipeline
- `plant_ml/models/flower_classifier_v2.pth` — model checkpoint
- `plant_ml/models/flower_classes.json` — class labels
- `localization.py` — UI and report localization support

## License

This project is intended for research and educational use. Review the repository configuration for any licensing requirements before commercial use.
