# Botanical Research Assistant

A Streamlit application that combines an optional, local five-class flower
image classifier with an independent botanical literature-search and
AI-assisted research workspace.

## Features

- Search botanical names and questions across Europe PMC and PubMed. The
  requested result count is configurable, results are deduplicated and ranked,
  and searches are cached.
- Resolve simple plant names against GBIF. Choose among exact matches for an
  ambiguous common name, or search an unresolved name as entered without
  assigning it a species. Verified scientific names and available synonyms
  can expand the search.
- Review source metadata, abstracts, identifiers, source pages, and listed
  full-text links. Missing abstracts and source outages are reported instead
  of replaced with invented material.
- Generate an AI-assisted literature review with references constructed from
  retrieved records, plus a paginated PDF. It is explicitly not peer-reviewed.
- Retrieve an original paper PDF only when a source lists a free PDF. The
  download is checked for an approved HTTPS host, redirect destination, size,
  PDF signature, and end marker. Otherwise, use the official source page.
- Optionally classify a flower photograph using the project's existing
  ResNet18 model.

## Requirements

- Python 3.10 or later.
- The dependencies listed in `requirements.txt`. PyTorch and torchvision
  wheels must be available for your Python version and Windows environment.
- An OpenAI API key to use AI extraction, research-plan generation, and review
  generation. Literature search and flower prediction do not require this key.
- Internet access for Europe PMC, PubMed, GBIF, and AI-provider requests.

## Windows PowerShell setup

From the repository directory:

```powershell
Set-Location "C:\path\to\Botanical_research_AI_Platform"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `OPENAI_API_KEY` in `.env` to your own key. Never commit `.env` or paste
the key into source code. The AI model defaults to `gpt-4o-mini`; optionally
set `BOTANICAL_AI_MODEL` to another model supported by your configured
provider. The app uses the OpenAI SDK's default endpoint.

Start the app:

```powershell
python -m streamlit run .\app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Use the workspaces

### Botanical research

Choose **Research workspace** without uploading an image. Enter a common or
scientific name, genus, species, or question about cultivation, disease,
genetics, medicinal plants, or growing conditions. Select a maximum of 5–30
candidate papers and run the search.

For a short plant name, the app attempts a GBIF match. If the common name has
multiple exact matches, select the intended scientific name before searching.
If a name cannot be verified, clarify it or explicitly search the text as
entered; no identity is guessed. Research searches are not limited to the
classifier's categories.

Expand a result to review its metadata, abstract, source page, and any
source-listed full text. Use **Verify and retrieve original PDF** when a free
PDF is listed. A paper without a verified accessible PDF remains available
through its official article page when one is supplied.

Generate the AI literature review after searching. Its claims are constrained
to retrieved bibliographic metadata and abstracts; treat it as a research aid,
check claims against the linked sources, and do not treat it as a
peer-reviewed or journal-ready manuscript. Download the generated review as
Markdown or PDF. When there are no results, the app can create a clearly
labeled no-evidence search summary, not a findings review.

### Flower identification

Upload a flower image to receive a prediction among exactly these five
trained classes:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

This classifier does **not** identify arbitrary plants or verify botanical
species. Its confidence warning and identification caveat are intended to
make that limitation clear. Use **Explore related research** to start a
separate research query from its predicted flower name; the query can be
edited or replaced.

## Data sources and limitations

- Literature search currently queries **Europe PMC** and **PubMed**. It does
  not cover every botanical, regional, or publisher repository.
- Short-name taxonomic matching and synonym lookup use **GBIF**. If GBIF is
  unavailable or a name is ambiguous, the app does not silently infer a
  species.
- Search completeness and relevance depend on source indexing, query wording,
  and network availability. A search with no results does not establish that
  no research exists.
- AI evidence extraction and synthesis are generally abstract-based; they are
  not a substitute for reviewing the original paper or its full text.
- Only free PDFs listed by the scientific source and returned from an
  approved source host are eligible for download. Paywalls are never bypassed.
- The classifier is a fixed five-class model, not a general plant
  identification system.

## Tests

From the project directory, run the offline unit and Streamlit UI tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v -p "test_*.py"
```

External literature, taxonomy, and PDF services are mocked in the tests. The
classifier tests run the existing checkpoint against a generated sample image.
Compile the Python sources with:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
```

## Project structure

- `app.py` — Streamlit interface and session workflow.
- `literature_search.py` — Europe PMC/PubMed retrieval, GBIF resolution, and
  guarded original-PDF downloads.
- `workflow.py` — literature search, abstract extraction, and plan workflow.
- `extraction_agent.py` and `protocol_generator.py` — AI evidence extraction,
  research-plan, and literature-review generation.
- `report_pdf.py` — multi-page review PDF rendering.
- `flower_classifier.py` — unchanged five-class ResNet18 inference pipeline.
- `plant_ml/models/flower_classifier_v2.pth` and
  `plant_ml/models/flower_classes.json` — classifier checkpoint and class map.
- `localization.py` — interface/report language support.
- `test_universal_research.py` — research, PDF, classifier, and Streamlit UI
  tests.
