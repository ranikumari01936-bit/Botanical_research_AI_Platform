import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


LANGUAGES = {
    "English": "English",
    "Arabic": "العربية",
    "Bengali": "বাংলা",
    "Chinese (Simplified)": "简体中文",
    "Chinese (Traditional)": "繁體中文",
    "Filipino": "Filipino",
    "French": "Français",
    "German": "Deutsch",
    "Hindi": "हिन्दी",
    "Indonesian": "Bahasa Indonesia",
    "Italian": "Italiano",
    "Japanese": "日本語",
    "Korean": "한국어",
    "Malay": "Bahasa Melayu",
    "Marathi": "मराठी",
    "Persian": "فارسی",
    "Polish": "Polski",
    "Portuguese": "Português",
    "Punjabi": "ਪੰਜਾਬੀ",
    "Russian": "Русский",
    "Spanish": "Español",
    "Swahili": "Kiswahili",
    "Tamil": "தமிழ்",
    "Telugu": "తెలుగు",
    "Thai": "ไทย",
    "Turkish": "Türkçe",
    "Urdu": "اردو",
    "Vietnamese": "Tiếng Việt",
    "Yoruba": "Yorùbá",
}

UI_TEXT = {
    "settings": "Settings",
    "appearance_and_language": "Appearance and language",
    "theme": "Color theme",
    "light": "Light",
    "dark": "Dark",
    "language": "Interface and report language",
    "language_help": "Choose a language for the app interface and translated research reports.",
    "translation_note": "Translation uses your configured AI provider. Review scientific terms and original sources.",
    "hero_kicker": "Field notes · AI research studio",
    "hero_title": "A clearer view of the botanical world.",
    "hero_description": "Identify flowers from a photograph, explore scientific literature, and turn published evidence into a research-ready plan.",
    "choose_workspace": "Choose a workspace",
    "flower_workspace": "Flower identification",
    "research_workspace": "Research workspace",
    "identify_flower": "Identify a flower",
    "flower_intro": "Upload a clear flower photo. The local image model will predict one of its supported flower categories.",
    "confidence_threshold": "Flag predictions below this confidence",
    "confidence_help": "Lower-confidence results are shown with a caution because the model may be uncertain.",
    "choose_flower_photo": "Choose a flower photo",
    "photo_help": "For the clearest result, use a well-lit photo with one flower in focus.",
    "empty_upload_title": "Your field specimen starts here",
    "empty_upload_body": "Add a JPG, PNG, or WebP image to see a preview and request an identification.",
    "large_image_error": "This image is larger than 10 MB. Please upload a smaller photo.",
    "specimen_ready": "Specimen ready",
    "image_alt": "Uploaded flower specimen",
    "pixel_unit": "pixels",
    "color_mode": "color mode",
    "identify_this_flower": "Identify this flower",
    "analyzing_image": "Analyzing the flower image...",
    "identification_result": "Identification result",
    "predicted_flower": "Predicted flower",
    "model_confidence": "Model confidence",
    "confidence_label": "Confidence",
    "low_confidence_warning": "This is a low-confidence prediction. Try a sharper, well-lit image with the flower filling more of the frame, or verify the identification with a botanical reference.",
    "confident_match": "The model found a reasonably confident match.",
    "prediction_caveat": "The model predicts among its trained flower categories. Treat this as a starting point, not a verified botanical identification.",
    "explore_research": "Explore related research",
    "unreadable_image": "That file could not be read as an image. Please try another photo.",
    "flower_failed": "Flower identification failed. Check the model setup and try again.",
    "research_title": "Botanical research",
    "research_intro": "Search Europe PMC and PubMed for any plant name or botanical research question. Research is independent from flower image classification.",
    "research_topic": "Plant or research topic",
    "research_placeholder": "e.g. Neem, Azadirachta indica, or drought resistance in rice",
    "research_help": "Enter a common or scientific plant name, genus, species, or botanical research question.",
    "candidate_papers": "Maximum candidate papers to retrieve",
    "search_and_plan": "Search literature and build a plan",
    "enter_topic": "Enter a plant or research topic to start the literature search.",
    "research_spinner": "Searching Europe PMC and PubMed, extracting abstracts, and drafting an evidence-led plan...",
    "research_failed": "The research workflow could not complete. Check your network and AI provider configuration, then try again.",
    "research_question": "Research question",
    "papers_found": "Papers found",
    "papers_analyzed": "Papers analyzed",
    "search_date": "Search date",
    "plan_status": "Research plan",
    "ready": "Ready",
    "unavailable": "Unavailable",
    "extraction_failed": "Could not extract evidence from",
    "evidence_scope": "Evidence extraction and the research plan use retrieved titles and abstracts, not full-text review. Relevant papers are not guaranteed; verify claims and citations against the linked sources.",
    "search_source_warning": "Some literature sources could not be reached. Results may be incomplete.",
    "taxon_ambiguous": "This common name matches more than one plant in GBIF. Please choose the intended scientific name before searching.",
    "taxon_unresolved": "GBIF could not confidently resolve this name. Clarify the plant (for example, add its scientific name), or explicitly search the text as written without assigning a species.",
    "taxon_unavailable": "The plant-name authority is unavailable. No species has been inferred; retry later or explicitly search the text as written.",
    "select_scientific_name": "Select the intended scientific name",
    "search_selected_taxon": "Search selected plant",
    "search_as_entered": "Search this text as entered (no species inferred)",
    "resolved_taxon": "Plant name matched in GBIF",
    "paper_heading": "Papers found",
    "no_papers": "No literature records were retrieved for this search.",
    "no_literature_results": "No records were returned by Europe PMC or PubMed. This does not prove that no relevant botanical research exists.",
    "abstract": "Abstract",
    "no_abstract": "No abstract was provided.",
    "authors": "Authors",
    "publication_year": "Publication year",
    "identifier": "Verified identifier",
    "open_source": "Open original source",
    "open_full_text": "Open available full text",
    "full_text_available": "Full-text access is listed by the source.",
    "full_text_unverified": "Full-text availability could not be verified from the returned source metadata.",
    "no_open_pdf": "No verified free PDF link was returned. Check the official source record for access information.",
    "pdf_downloading": "Verifying and retrieving the original paper PDF from its official source...",
    "check_original_pdf": "Verify and retrieve original PDF",
    "download_original_pdf": "Download original published paper PDF",
    "pdf_failed": "The source did not return a valid PDF. Use the official article page instead.",
    "plan_generation_failed": "The papers were retrieved, but the AI research-plan draft could not be generated.",
    "generate_literature_review": "Generate AI literature review",
    "generate_no_evidence_report": "Create no-evidence search summary",
    "literature_review_title": "AI-generated literature review",
    "review_spinner": "Synthesizing retrieved source records into an evidence-limited literature review...",
    "review_failed": "The AI literature review could not be generated. Check the AI provider configuration and try again.",
    "review_pdf_failed": "The review is available above, but its PDF could not be generated. Try the PDF download again.",
    "download_review_pdf": "Download AI-generated literature review PDF",
    "download_review_markdown": "Download literature review as Markdown",
    "retry_review_pdf": "Retry PDF generation",
    "review_notice": "AI-generated literature review based on retrieved records; not a peer-reviewed publication and not a journal-ready manuscript.",
    "no_evidence_report_notice": "No-evidence search summary only; no literature findings were generated.",
    "evidence_extraction": "AI evidence extraction",
    "plan_heading": "Research plan",
    "download_report": "Download research report",
    "translate_report": "Translate full report",
    "translate_spinner": "Translating the research report...",
    "translated_report": "Translated research report",
    "pmid_label": "PMID",
    "journal_label": "Journal",
    "view_pubmed_source": "View source on PubMed",
    "not_available": "Not available",
    "not_listed": "Not listed",
    "translation_failed": "Could not translate the report. Check the AI provider configuration or network and try again.",
    "no_plan_no_papers": "No research plan was generated because no papers were found.",
    "no_plan_no_evidence": "No research plan was generated because evidence could not be extracted from the papers.",
    "generated_translation_notice": "AI-generated translation. Verify scientific terminology against the original paper and report.",
}


def _client() -> OpenAI:
    load_dotenv()
    return OpenAI(timeout=60.0, max_retries=1)


def _model_name() -> str:
    return os.getenv("BOTANICAL_AI_MODEL", "gpt-4o-mini")


@st.cache_data(ttl="30d", max_entries=40)
def translate_interface(language: str) -> dict[str, str]:
    if language == "English":
        return {}
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported interface language: {language}")

    response = _client().responses.create(
        model=_model_name(),
        input=(
            f"Translate every value in this JSON object into {language}. "
            "Return a JSON object with exactly the same keys. Keep each value "
            "concise and suitable for a user interface. Preserve placeholders, "
            "HTML-free plain text, file extensions, and scientific meaning. "
            "Return only valid JSON.\n\n"
            f"{json.dumps(UI_TEXT, ensure_ascii=False)}"
        ),
    )
    translations = json.loads(response.output_text)

    if not isinstance(translations, dict) or set(translations) != set(UI_TEXT):
        raise ValueError("The translation response did not include all UI strings.")
    if not all(isinstance(value, str) and value.strip() for value in translations.values()):
        raise ValueError("The translation response contains an empty UI string.")

    return translations


def translate_report(report: str, language: str) -> str:
    if language == "English":
        return report
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported report language: {language}")
    if not report.strip():
        raise ValueError("There is no research report to translate.")

    response = _client().responses.create(
        model=_model_name(),
        input=(
            f"Translate the complete botanical research report below into {language}. "
            "Preserve Markdown structure, paper titles, author names, journal names, "
            "PMIDs, DOI strings, URLs, numbers, units, and citations exactly. "
            "Translate the scientific narrative faithfully without adding, removing, "
            "or correcting claims. Keep uncertainty and evidence limitations explicit. "
            "Return only the translated report.\n\n"
            f"{report}"
        ),
    )
    translated = response.output_text.strip()
    if not translated:
        raise ValueError("The translation provider returned an empty report.")
    return translated
