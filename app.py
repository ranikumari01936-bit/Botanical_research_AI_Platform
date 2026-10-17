import hashlib
import html
import io
import logging
import re

import streamlit as st
from PIL import Image, UnidentifiedImageError

from flower_classifier import predict_flower
from literature_search import (
    LiteratureSearchError,
    download_original_pdf,
    get_taxon_synonyms,
    resolve_taxon,
)
from localization import LANGUAGES, UI_TEXT, translate_interface, translate_report


logger = logging.getLogger(__name__)
LOW_CONFIDENCE_THRESHOLD = 60
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


def flower_research_question(flower_name: str) -> str:
    flower = flower_name.lower()
    if flower.endswith("ies"):
        plural_flower = flower
    elif flower.endswith("y") and len(flower) > 1 and flower[-2] not in "aeiou":
        plural_flower = f"{flower[:-1]}ies"
    elif flower.endswith("s"):
        plural_flower = flower
    elif flower.endswith(("x", "z", "ch", "sh")):
        plural_flower = f"{flower}es"
    else:
        plural_flower = f"{flower}s"

    return (
        "What are the botanical characteristics, growing conditions, and "
        f"scientific research findings related to {plural_flower}?"
    )


def open_research_workspace(flower_name: str) -> None:
    st.session_state["workspace"] = "Research workspace"
    st.session_state["research_topic"] = flower_research_question(flower_name)
    st.session_state["research_result"] = None
    st.session_state["translated_report"] = None
    st.session_state["literature_review"] = None
    st.session_state["literature_review_pdf"] = None
    st.session_state["original_pdfs"] = {}
    st.session_state["original_pdf_errors"] = {}


def start_research(
    question: str,
    max_results: int,
    taxonomy: dict | None = None,
) -> None:
    taxonomy = dict(taxonomy or {})
    if taxonomy.get("gbif_key") and "synonyms" not in taxonomy:
        try:
            taxonomy["synonyms"] = get_taxon_synonyms(taxonomy["gbif_key"])
        except Exception:
            logger.warning(
                "GBIF synonyms could not be retrieved for taxon %s",
                taxonomy["gbif_key"],
                exc_info=True,
            )
            taxonomy["synonym_warning"] = (
                "GBIF synonyms could not be retrieved; the search used the "
                "entered name and verified scientific name only."
            )

    st.session_state["research_query"] = question
    st.session_state["research_result"] = None
    st.session_state["translated_report"] = None
    st.session_state["literature_review"] = None
    st.session_state["literature_review_pdf"] = None
    st.session_state["original_pdfs"] = {}
    st.session_state["original_pdf_errors"] = {}

    workflow_input = {
        "question": question,
        "max_results": max_results,
        "scientific_name": (
            taxonomy.get("canonical_name", "") if taxonomy else ""
        ),
        "taxonomy": taxonomy or {},
    }
    try:
        with st.spinner(text("research_spinner")):
            from workflow import app as research_workflow

            result = research_workflow.invoke(workflow_input)
        st.session_state["research_result"] = result
    except LiteratureSearchError as error:
        logger.warning("Literature search could not complete: %s", error)
        st.error(str(error))
    except Exception:
        logger.exception("Botanical research workflow failed")
        st.error(text("research_failed"))


def generate_review_pdf(report: str, question: str) -> None:
    try:
        from report_pdf import generate_report_pdf

        result = st.session_state.get("research_result") or {}
        has_sources = bool(result.get("papers"))
        pdf_bytes = generate_report_pdf(
            report,
            title=(
                f"Botanical literature review: {question[:100]}"
                if has_sources
                else "Botanical no-evidence search summary"
            ),
            notice=(
                text("review_notice")
                if has_sources
                else text("no_evidence_report_notice")
            ),
        )
        st.session_state["literature_review_pdf"] = {
            "question": question,
            "content": pdf_bytes,
        }
    except Exception:
        logger.exception("AI literature review PDF generation failed")
        st.session_state["literature_review_pdf"] = None
        st.error(text("review_pdf_failed"))


def build_research_report(result: dict, question: str) -> str:
    sections = [
        f"# Botanical research report: {question}",
        "AI-generated evidence summary and research-plan draft; not a "
        "peer-reviewed publication.",
        f"## Research question\n{question}",
        (
            "## Evidence scope\n"
            "This report summarizes Europe PMC and PubMed records retrieved "
            "for this query. "
            "AI evidence extraction and the research-plan draft are based on "
            "the retrieved titles and abstracts, not full-text review. "
            "Searches may return no relevant papers; findings and citations "
            "should be checked against the linked original sources."
        ),
        f"## Search date\n{result.get('search_date') or 'Not available'}",
        (
            "## Search methods\n"
            f"Sources queried: {', '.join(result.get('search_sources', [])) or 'Not listed'}. "
            f"Scientific-name and synonym terms: "
            f"{', '.join(result.get('searched_taxon_terms', [])) or 'None verified'}."
        ),
    ]

    taxonomy = result.get("taxonomy") or {}
    if taxonomy.get("canonical_name") or taxonomy.get("scientific_name"):
        scientific_name = (
            taxonomy.get("canonical_name") or taxonomy.get("scientific_name")
        )
        taxon_source = taxonomy.get("source_url") or "GBIF"
        sections.append(
            f"## Verified taxonomy\n{scientific_name} "
            f"(source: {taxon_source})."
        )

    for warning in result.get("search_warnings", []):
        sections.append(f"**Search limitation:** {warning}")

    extracted_by_id = {
        str(item.get("pmid") or item.get("pmcid") or item.get("doi") or ""): item
        for item in result.get("extracted_information", [])
    }
    sections.append("## Literature search results")
    for paper in result.get("papers", []):
        identifiers = [
            f"PMID: {paper['pmid']}" if paper.get("pmid") else "",
            f"PMCID: {paper['pmcid']}" if paper.get("pmcid") else "",
            f"DOI: {paper['doi']}" if paper.get("doi") else "",
        ]
        sections.extend([
            f"### {paper.get('title') or 'Untitled paper'}",
            f"- Authors: {paper.get('authors') or 'Not listed'}",
            f"- Publication year: {paper.get('pub_year') or 'Not listed'}",
            f"- Journal: {paper.get('journal') or 'Not listed'}",
            f"- Identifiers: {', '.join(value for value in identifiers if value) or 'Not available'}",
            f"**Abstract**\n\n{paper.get('abstract') or 'No abstract was provided.'}",
        ])
        if paper.get("source_url"):
            sections.append(f"- Original source: {paper['source_url']}")
        extracted = extracted_by_id.get(str(
            paper.get("pmid") or paper.get("pmcid") or paper.get("doi") or ""
        ))
        if extracted:
            sections.append(
                f"**AI evidence extraction**\n\n"
                f"{extracted.get('extracted', '')}"
            )

    if not result.get("papers"):
        sections.append(
            "No records were returned by Europe PMC or PubMed. This does "
            "not prove that no relevant "
            "research exists."
        )
    sections.extend([
        "## Key findings and research plan",
        result.get("final_plan") or "No evidence-based research plan was generated.",
    ])
    return "\n\n".join(sections)


st.set_page_config(
    page_title="Botanical Research Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.session_state.setdefault("theme_mode", "light")
st.session_state.setdefault("report_language", "English")
st.session_state.setdefault("workspace", "Flower identification")
st.session_state.setdefault("flower_result", None)
st.session_state.setdefault("research_result", None)
st.session_state.setdefault("research_query", "")
st.session_state.setdefault("research_topic", "")
st.session_state.setdefault("translated_report", None)
st.session_state.setdefault("pending_taxon", None)
st.session_state.setdefault("literature_review", None)
st.session_state.setdefault("literature_review_pdf", None)
st.session_state.setdefault("original_pdfs", {})
st.session_state.setdefault("original_pdf_errors", {})
st.session_state.setdefault("ui_translation_error", None)

selected_language = st.session_state["report_language"]
translations: dict[str, str] = {}
if selected_language != "English":
    try:
        translations = translate_interface(selected_language)
        st.session_state["ui_translation_error"] = None
    except Exception:
        logger.exception("Interface translation failed for %s", selected_language)
        st.session_state["ui_translation_error"] = selected_language


def text(key: str) -> str:
    return translations.get(key, UI_TEXT[key])


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(ellipse at 8% 2%, rgba(185, 216, 174, .30), transparent 28rem),
                radial-gradient(ellipse at 94% 20%, rgba(226, 231, 192, .28), transparent 26rem),
                linear-gradient(145deg, #f7faf5 0%, #f3f7f1 48%, #f8f8f2 100%);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            padding-top: 3.5rem;
            padding-bottom: 4rem;
            overflow: visible;
        }
        .botanical-hero {
            position: relative;
            isolation: isolate;
            margin-top: 0.5rem;
            margin-bottom: 1.75rem;
            padding: clamp(1.75rem, 4vw, 3.25rem);
            overflow: hidden;
            border: 1px solid rgba(57, 92, 67, .14);
            border-radius: 28px;
            background:
                radial-gradient(ellipse at 88% 18%, rgba(191, 216, 170, .38), transparent 30%),
                linear-gradient(120deg, rgba(255, 255, 255, .88), rgba(239, 246, 235, .76));
            box-shadow: 0 18px 55px rgba(41, 69, 45, .08);
            backdrop-filter: blur(18px);
        }
        .botanical-hero::after {
            position: absolute;
            top: -4rem;
            right: 5%;
            width: 18rem;
            height: 18rem;
            border: 1px solid rgba(74, 112, 76, .10);
            border-radius: 68% 32% 65% 35% / 40% 55% 45% 60%;
            background:
                repeating-linear-gradient(135deg, transparent 0 13px, rgba(74, 112, 76, .035) 14px 15px);
            content: "";
            transform: rotate(-16deg);
            pointer-events: none;
        }
        .botanical-kicker {
            margin: 0 0 .75rem;
            color: #4b7657;
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .botanical-title {
            position: relative;
            z-index: 1;
            display: block;
            width: 100%;
            max-width: 760px;
            margin: 0;
            padding-top: 0.08em;
            padding-bottom: 0.06em;
            color: #20392b;
            font-size: clamp(2rem, 5vw, 3.7rem);
            font-weight: 650;
            letter-spacing: -.045em;
            line-height: 1.18;
            overflow-wrap: anywhere;
        }
        .botanical-description {
            position: relative;
            z-index: 1;
            max-width: 690px;
            margin: 1rem 0 0;
            color: #5b6c5e;
            font-size: 1.05rem;
            line-height: 1.65;
        }
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stForm"],
        [data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, .82) !important;
            border-radius: 22px !important;
            background: rgba(255, 255, 255, .58) !important;
            box-shadow:
                0 12px 34px rgba(44, 75, 49, .07),
                inset 0 1px 0 rgba(255, 255, 255, .78);
            backdrop-filter: blur(18px) saturate(135%);
            -webkit-backdrop-filter: blur(18px) saturate(135%);
        }
        [data-testid="stForm"] {
            padding: 1.15rem 1.25rem !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: inherit;
        }
        [data-testid="stExpander"] {
            margin-bottom: .65rem;
            transition: transform .18s ease, box-shadow .18s ease;
        }
        [data-testid="stExpander"]:hover {
            transform: translateY(-2px);
            box-shadow:
                0 16px 38px rgba(44, 75, 49, .11),
                inset 0 1px 0 rgba(255, 255, 255, .85);
        }
        [data-testid="stMetric"] {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(255, 255, 255, .84);
            border-radius: 18px;
            background: rgba(255, 255, 255, .55);
            box-shadow: 0 10px 28px rgba(44, 75, 49, .06);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }
        .st-key-flower-result-card {
            padding: clamp(1rem, 3vw, 1.5rem);
            border: 1px solid rgba(57, 92, 67, .16);
            border-radius: 24px;
            background:
                radial-gradient(ellipse at 92% 8%, rgba(191, 216, 170, .24), transparent 36%),
                linear-gradient(135deg, rgba(255, 255, 255, .78), rgba(239, 246, 235, .68));
            box-shadow: 0 16px 42px rgba(41, 69, 45, .09);
            backdrop-filter: blur(18px) saturate(130%);
            -webkit-backdrop-filter: blur(18px) saturate(130%);
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed rgba(71, 123, 85, .42);
            border-radius: 18px;
            background: rgba(255, 255, 255, .48);
            transition: background .18s ease, border-color .18s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(71, 123, 85, .72);
            background: rgba(255, 255, 255, .72);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border-color: rgba(71, 123, 85, .22);
            border-radius: 14px;
            background: rgba(255, 255, 255, .58);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }
        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 14px;
            transition: transform .18s ease, box-shadow .18s ease;
        }
        [data-testid="stButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(55, 105, 66, .18);
        }
        [data-testid="stButtonGroup"] {
            width: fit-content;
            max-width: 100%;
            margin: .5rem 0 1.5rem;
            padding: .35rem;
            border: 1px solid rgba(255, 255, 255, .9);
            border-radius: 18px;
            background: rgba(255, 255, 255, .54);
            box-shadow: 0 8px 24px rgba(44, 75, 49, .07);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }
        [data-testid="stButtonGroup"] button {
            min-height: 44px;
            padding-inline: 1rem;
            border-radius: 13px;
            white-space: normal;
            line-height: 1.25;
        }
        [data-testid="stButton"] button:focus-visible,
        [data-testid="stFormSubmitButton"] button:focus-visible,
        [data-testid="stButtonGroup"] button:focus-visible,
        [data-testid="stFileUploaderDropzone"] button:focus-visible,
        [data-testid="stTextInput"] input:focus-visible {
            outline: 3px solid rgba(71, 123, 85, .5);
            outline-offset: 2px;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] p,
        [data-testid="stMarkdownContainer"] {
            overflow-wrap: anywhere;
        }
        [data-testid="stProgress"] > div > div {
            border-radius: 999px;
        }
        @media (prefers-reduced-motion: reduce) {
            [data-testid="stExpander"],
            [data-testid="stButton"] button,
            [data-testid="stFormSubmitButton"] button {
                transition: none;
            }
        }
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 2rem;
            }
            .botanical-hero {
                border-radius: 20px;
                padding: 1.25rem;
            }
            .botanical-title {
                font-size: clamp(1.85rem, 8vw, 2.5rem);
                letter-spacing: -.035em;
            }
            .botanical-description {
                font-size: .98rem;
                line-height: 1.55;
            }
            [data-testid="stButtonGroup"] {
                display: flex;
                width: 100%;
                gap: .25rem;
                padding: .3rem;
            }
            [data-testid="stButtonGroup"] button {
                flex: 1 1 0;
                min-width: 0;
                min-height: 48px;
                padding-inline: .45rem;
                font-size: .88rem;
            }
            [data-testid="stForm"] {
                padding: .9rem !important;
            }
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stForm"],
            [data-testid="stExpander"] {
                border-radius: 17px !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state["theme_mode"] == "dark":
    st.markdown(
        """
        <style>
            :root {
                --background-color: #111a15;
                --secondary-background-color: #1b2920;
                --text-color: #e7eee5;
                --primary-color: #91bd91;
                color-scheme: dark;
            }
            .stApp,
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(ellipse at 8% 2%, rgba(79, 117, 79, .24), transparent 28rem),
                    radial-gradient(ellipse at 94% 20%, rgba(76, 95, 59, .18), transparent 26rem),
                    linear-gradient(145deg, #111a15 0%, #17231b 48%, #111a15 100%);
                color: #e7eee5;
            }
            .botanical-hero {
                border-color: rgba(220, 239, 218, .12);
                background:
                    radial-gradient(ellipse at 88% 18%, rgba(93, 135, 87, .23), transparent 30%),
                    linear-gradient(120deg, rgba(35, 52, 41, .84), rgba(27, 43, 33, .76));
                box-shadow: 0 18px 55px rgba(0, 0, 0, .2);
            }
            .botanical-title { color: #edf4e9; }
            .botanical-kicker { color: #a5c9a4; }
            .botanical-description { color: #c1d0bf; }
            .st-key-flower-result-card {
                border-color: rgba(220, 239, 218, .14);
                background:
                    radial-gradient(ellipse at 92% 8%, rgba(93, 135, 87, .2), transparent 36%),
                    linear-gradient(135deg, rgba(35, 52, 41, .84), rgba(27, 43, 33, .76));
                box-shadow: 0 16px 42px rgba(0, 0, 0, .2);
            }
            .stApp {
                --background-color: #111a15;
                --secondary-background-color: #1b2920;
                --text-color: #e7eee5;
                --primary-color: #91bd91;
            }
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stForm"],
            [data-testid="stExpander"],
            [data-testid="stMetric"],
            [data-testid="stButtonGroup"],
            [data-testid="stFileUploaderDropzone"] {
                border-color: rgba(219, 238, 215, .14) !important;
                background: rgba(29, 45, 35, .68) !important;
                box-shadow:
                    0 12px 34px rgba(0, 0, 0, .16),
                    inset 0 1px 0 rgba(238, 249, 235, .06);
            }
            [data-testid="stTextInput"] input,
            [data-testid="stSelectbox"] [data-baseweb="select"] > div {
                border-color: rgba(194, 222, 187, .22);
                background: rgba(25, 40, 31, .8);
                color: #edf4e9;
            }
            [data-testid="stTextInput"] input::placeholder {
                color: #a7b6a5;
            }
            [data-testid="stButtonGroup"] button { color: #dfeadd; }
            [data-testid="stButtonGroup"] button[aria-checked="false"] {
                background: rgba(29, 45, 35, .92);
                color: #dfeadd;
            }
            [data-testid="stButtonGroup"] button[aria-checked="true"] {
                border-color: #6c9d70 !important;
                background: #3c6847 !important;
                color: #fff !important;
            }
            [data-testid="stPopoverButton"] {
                border-color: rgba(219, 238, 215, .22);
                background: rgba(29, 45, 35, .92);
                color: #edf4e9;
            }
            [data-testid="stFileUploaderDropzone"] { color: #dfeadd; }
            [data-testid="stCaption"],
            [data-testid="stWidgetLabel"] { color: #bdcdbb; }
            [data-testid="stPopover"] > div,
            [data-testid="stPopoverBody"] {
                border-color: rgba(219, 238, 215, .14);
                background: #1b2920;
                color: #e7eee5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.container(horizontal=True, horizontal_alignment="right"):
    with st.popover(text("settings"), icon=":material/settings:"):
        st.subheader(text("appearance_and_language"))
        st.selectbox(
            text("theme"),
            options=["light", "dark"],
            format_func=lambda value: text(value),
            key="theme_mode",
        )
        st.selectbox(
            text("language"),
            options=list(LANGUAGES),
            format_func=lambda language: f"{language} — {LANGUAGES[language]}",
            help=text("language_help"),
            key="report_language",
        )
        st.caption(text("translation_note"))

if st.session_state["ui_translation_error"]:
    st.error(
        f"Could not translate the interface to "
        f"{st.session_state['ui_translation_error']}. "
        "The app is shown in English; check AI provider settings and try again."
    )

st.markdown(
    f"""
    <section class="botanical-hero">
        <p class="botanical-kicker">{html.escape(text("hero_kicker"))}</p>
        <h1 class="botanical-title">{html.escape(text("hero_title"))}</h1>
        <p class="botanical-description">
            {html.escape(text("hero_description"))}
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

workspace = st.segmented_control(
    text("choose_workspace"),
    options=["Flower identification", "Research workspace"],
    format_func=lambda value: text(
        "flower_workspace" if value == "Flower identification" else "research_workspace"
    ),
    label_visibility="collapsed",
    key="workspace",
)

if workspace == "Flower identification":
    st.header(text("identify_flower"), icon=":material/local_florist:")
    st.caption(text("flower_intro"))

    confidence_threshold = st.slider(
        text("confidence_threshold"),
        min_value=40,
        max_value=90,
        value=LOW_CONFIDENCE_THRESHOLD,
        step=5,
        format="%d%%",
        help=text("confidence_help"),
    )

    uploaded_image = st.file_uploader(
        text("choose_flower_photo"),
        type=["jpg", "jpeg", "png", "webp"],
        help=text("photo_help"),
    )

    if uploaded_image is None:
        with st.container(border=True):
            st.markdown(f"**{text('empty_upload_title')}**")
            st.write(text("empty_upload_body"))
    else:
        image_bytes = uploaded_image.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()

        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            st.error(text("large_image_error"))
        else:
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    preview_column, details_column = st.columns(
                        [1.05, 1],
                        vertical_alignment="center",
                    )
                    with preview_column:
                        st.image(
                            image,
                            caption=uploaded_image.name,
                            alt=text("image_alt"),
                            width="stretch",
                        )
                    with details_column:
                        st.markdown(f"#### {text('specimen_ready')}")
                        st.caption(
                            f"{image.width} × {image.height} {text('pixel_unit')} · "
                            f"{image.mode} {text('color_mode')}"
                        )
                        analyze = st.button(
                            text("identify_this_flower"),
                            type="primary",
                            icon=":material/search:",
                            width="stretch",
                        )

                        if analyze:
                            st.session_state["flower_result"] = None
                            with st.spinner(text("analyzing_image")):
                                st.session_state["flower_result"] = {
                                    "digest": image_digest,
                                    "prediction": predict_flower(image.copy()),
                                }

                flower_result = st.session_state.get("flower_result")
                if flower_result and flower_result["digest"] == image_digest:
                    prediction = flower_result["prediction"]
                    flower_name = prediction["flower"].replace("_", " ").title()
                    confidence = float(prediction["confidence"])

                    with st.container(key="flower-result-card"):
                        st.subheader(text("identification_result"))
                        specimen_column, result_column = st.columns(
                            [0.9, 1.1],
                            vertical_alignment="center",
                        )
                        with specimen_column:
                            st.image(
                                image_bytes,
                                caption=uploaded_image.name,
                                alt=text("image_alt"),
                                width="stretch",
                            )
                        with result_column:
                            st.metric(text("predicted_flower"), flower_name)
                            st.metric(
                                text("model_confidence"),
                                f"{confidence:.1f}%",
                            )
                            st.progress(
                                max(0.0, min(confidence / 100, 1.0)),
                                text=(
                                    f"{text('confidence_label')}: "
                                    f"{confidence:.1f}%"
                                ),
                            )

                            if confidence < confidence_threshold:
                                st.warning(text("low_confidence_warning"))
                            else:
                                st.success(text("confident_match"))
                            st.caption(text("prediction_caveat"))
                            st.button(
                                text("explore_research"),
                                type="primary",
                                icon=":material/biotech:",
                                width="stretch",
                                on_click=open_research_workspace,
                                args=(flower_name,),
                            )
            except (UnidentifiedImageError, OSError, ValueError) as error:
                logger.warning("Uploaded flower image could not be processed: %s", error)
                st.error(text("unreadable_image"))
            except Exception as error:
                logger.exception("Flower identification failed")
                st.error(text("flower_failed"))

else:
    st.header(text("research_title"), icon=":material/biotech:")
    st.write(text("research_intro"))

    with st.form("research_search", border=True):
        research_query = st.text_input(
            text("research_topic"),
            placeholder=text("research_placeholder"),
            help=text("research_help"),
            key="research_topic",
        )
        max_results = st.select_slider(
            text("candidate_papers"),
            options=[5, 10, 15, 20, 30],
            value=10,
        )
        run_research = st.form_submit_button(
            text("search_and_plan"),
            type="primary",
            icon=":material/search:",
        )

    if run_research:
        st.session_state["pending_taxon"] = None
        clean_query = research_query.strip()
        if not clean_query:
            st.warning(text("enter_topic"))
        else:
            try:
                resolution = resolve_taxon(clean_query)
            except Exception:
                logger.exception("Botanical name resolution failed")
                resolution = {
                    "status": "unavailable",
                    "query": clean_query,
                    "candidates": [],
                }

            if resolution["status"] == "resolved":
                taxonomy = resolution["candidates"][0]
                taxonomy["source"] = resolution["source"]
                start_research(clean_query, max_results, taxonomy)
            elif resolution["status"] in {"ambiguous", "unresolved", "unavailable"}:
                st.session_state["pending_taxon"] = {
                    "question": clean_query,
                    "max_results": max_results,
                    "resolution": resolution,
                }
                st.session_state["research_result"] = None
                st.session_state["literature_review"] = None
                st.session_state["literature_review_pdf"] = None
            else:
                st.session_state["pending_taxon"] = None
                start_research(clean_query, max_results)

    pending_taxon = st.session_state.get("pending_taxon")
    if pending_taxon:
        resolution = pending_taxon["resolution"]
        candidates = resolution.get("candidates", [])
        if resolution["status"] == "ambiguous":
            st.warning(text("taxon_ambiguous"))
            selected_taxon = st.selectbox(
                text("select_scientific_name"),
                options=candidates,
                format_func=lambda candidate: (
                    f"{candidate['canonical_name']} · {candidate['rank']}"
                ),
                key="pending_taxon_choice",
            )
            if st.button(
                text("search_selected_taxon"),
                type="primary",
                key="search_selected_taxon",
            ):
                taxonomy = dict(selected_taxon)
                taxonomy["source"] = resolution["source"]
                st.session_state["pending_taxon"] = None
                start_research(
                    pending_taxon["question"],
                    pending_taxon["max_results"],
                    taxonomy,
                )
        else:
            st.info(
                text(
                    "taxon_unavailable"
                    if resolution["status"] == "unavailable"
                    else "taxon_unresolved"
                )
            )
            if st.button(
                text("search_as_entered"),
                key="search_taxon_as_entered",
            ):
                st.session_state["pending_taxon"] = None
                start_research(
                    pending_taxon["question"],
                    pending_taxon["max_results"],
                )

    result = st.session_state.get("research_result")
    if result:
        papers = result.get("papers", [])
        extracted_information = result.get("extracted_information", [])
        extraction_errors = result.get("extraction_errors", [])

        st.caption(
            f"{text('research_question')} · {st.session_state['research_query']}"
        )
        if result.get("search_date"):
            st.caption(
                f"{text('search_date')}: {result['search_date']} · "
                f"{', '.join(result.get('search_sources', []))}"
            )
        taxonomy = result.get("taxonomy") or {}
        if taxonomy.get("canonical_name"):
            st.caption(
                f"{text('resolved_taxon')}: {taxonomy['canonical_name']} · "
                f"{taxonomy.get('source') or 'GBIF'}"
            )
        for search_warning in result.get("search_warnings", []):
            st.warning(search_warning)

        metric_papers, metric_extracted, metric_plan = st.columns(3)
        metric_papers.metric(text("papers_found"), len(papers))
        metric_extracted.metric(text("papers_analyzed"), len(extracted_information))
        metric_plan.metric(
            text("plan_status"),
            text("ready") if result.get("final_plan") else text("unavailable"),
        )

        if extraction_errors:
            for extraction_error in extraction_errors:
                st.warning(
                    f"{text('extraction_failed')} "
                    f"**{extraction_error['title']}**: {extraction_error['error']}"
                )

        st.caption(text("evidence_scope"))
        st.subheader(text("paper_heading"), icon=":material/menu_book:")
        if not papers:
            st.info(text("no_literature_results"))
        else:
            extracted_by_id = {
                str(item.get("pmid") or item.get("pmcid") or item.get("doi") or ""):
                item
                for item in extracted_information
            }
            for paper_index, paper in enumerate(papers):
                title = paper.get("title") or "Untitled paper"
                with st.expander(title):
                    author_line = paper.get("authors") or text("not_available")
                    year = paper.get("pub_year") or text("not_available")
                    journal = paper.get("journal") or text("not_listed")
                    st.caption(
                        f"{text('authors')}: {author_line} · "
                        f"{text('publication_year')}: {year} · "
                        f"{text('journal_label')}: {journal}"
                    )

                    identifiers = [
                        f"{text('pmid_label')}: {paper['pmid']}"
                        for _ in [0]
                        if paper.get("pmid")
                    ] + [
                        f"PMCID: {paper['pmcid']}"
                        for _ in [0]
                        if paper.get("pmcid")
                    ] + [
                        f"DOI: {paper['doi']}"
                        for _ in [0]
                        if paper.get("doi")
                    ]
                    st.caption(
                        f"{text('identifier')}: "
                        f"{'; '.join(identifiers) or text('not_available')}"
                    )
                    source_url = paper.get("source_url") or ""
                    if source_url.startswith("https://"):
                        st.link_button(
                            text("open_source"),
                            source_url,
                            icon=":material/open_in_new:",
                            key=f"source-{paper_index}",
                        )
                    for full_text_index, full_text_url in enumerate(
                        paper.get("full_text_urls", [])
                    ):
                        if (
                            full_text_url.get("style") == "html"
                            and full_text_url.get("availability", "").lower()
                            in {"free", "open access", "unverified"}
                        ):
                            st.link_button(
                                text("open_full_text"),
                                full_text_url["url"],
                                icon=":material/open_in_new:",
                                key=f"full-text-{paper_index}-{full_text_index}",
                            )
                    free_full_text_listed = any(
                        str(item.get("availability") or "").casefold()
                        in {"free", "open access"}
                        for item in paper.get("full_text_urls", [])
                    )
                    if paper.get("is_open_access") or free_full_text_listed:
                        st.caption(text("full_text_available"))
                    else:
                        st.caption(text("full_text_unverified"))

                    st.markdown(f"**{text('abstract')}**")
                    st.write(paper.get("abstract") or text("no_abstract"))

                    identifier = str(
                        paper.get("pmid")
                        or paper.get("pmcid")
                        or paper.get("doi")
                        or ""
                    )
                    extracted = extracted_by_id.get(identifier)
                    if extracted:
                        st.markdown(f"**{text('evidence_extraction')}**")
                        st.markdown(extracted.get("extracted", ""))

                    paper_identity = identifier or f"record-{paper_index}"
                    state_key = hashlib.sha256(
                        paper_identity.encode("utf-8")
                    ).hexdigest()[:16]
                    pdf_state = st.session_state["original_pdfs"].get(state_key)
                    if pdf_state:
                        safe_filename = re.sub(
                            r"[^A-Za-z0-9_.-]+",
                            "-",
                            paper_identity,
                        )[:80]
                        st.download_button(
                            text("download_original_pdf"),
                            data=pdf_state,
                            file_name=f"original-paper-{safe_filename}.pdf",
                            mime="application/pdf",
                            key=f"original-pdf-{state_key}",
                            icon=":material/download:",
                        )
                    elif paper.get("pdf_urls"):
                        if st.button(
                            text("check_original_pdf"),
                            key=f"check-pdf-{state_key}",
                        ):
                            with st.spinner(text("pdf_downloading")):
                                pdf_bytes, pdf_error = download_original_pdf(paper)
                            if pdf_bytes:
                                st.session_state["original_pdfs"][state_key] = (
                                    pdf_bytes
                                )
                                st.session_state["original_pdf_errors"].pop(
                                    state_key,
                                    None,
                                )
                                st.rerun()
                            else:
                                st.session_state["original_pdf_errors"][
                                    state_key
                                ] = pdf_error
                        pdf_error = st.session_state["original_pdf_errors"].get(
                            state_key
                        )
                        if pdf_error:
                            st.warning(f"{text('pdf_failed')} {pdf_error}")
                    else:
                        st.caption(text("no_open_pdf"))

        final_plan = result.get("final_plan", "")
        if final_plan:
            st.subheader(text("plan_heading"), icon=":material/eco:")
            with st.container(border=True):
                st.markdown(final_plan)
        elif not papers:
            st.warning(text("no_plan_no_papers"))
        elif result.get("plan_error"):
            st.warning(
                f"{text('plan_generation_failed')} {result['plan_error']}"
            )
        else:
            st.warning(text("no_plan_no_evidence"))

        report_text = build_research_report(
            result,
            st.session_state["research_query"],
        )
        st.download_button(
            text("download_report"),
            data=report_text,
            file_name="botanical-research-report-en.md",
            mime="text/markdown",
            icon=":material/download:",
            key="download_english_report",
        )

        review_action = st.button(
            text(
                "generate_literature_review"
                if papers
                else "generate_no_evidence_report"
            ),
            type="primary",
            icon=":material/article:",
            key="generate_botanical_review",
        )
        if review_action:
            try:
                if papers:
                    with st.spinner(text("review_spinner")):
                        from protocol_generator import generate_literature_review

                        review = generate_literature_review(
                            question=st.session_state["research_query"],
                            papers=papers,
                            extracted_information=extracted_information,
                            search_date=result.get("search_date", ""),
                            taxonomy=result.get("taxonomy"),
                        )
                else:
                    from protocol_generator import generate_literature_review

                    review = generate_literature_review(
                        question=st.session_state["research_query"],
                        papers=[],
                        extracted_information=[],
                        search_date=result.get("search_date", ""),
                        taxonomy=result.get("taxonomy"),
                    )
                st.session_state["literature_review"] = {
                    "question": st.session_state["research_query"],
                    "content": review,
                    "ai_generated": bool(papers),
                }
                generate_review_pdf(
                    review,
                    st.session_state["research_query"],
                )
            except Exception:
                logger.exception("AI botanical literature review failed")
                st.error(text("review_failed"))

        literature_review = st.session_state.get("literature_review")
        if (
            literature_review
            and literature_review.get("question")
            == st.session_state["research_query"]
        ):
            st.subheader(text("literature_review_title"), icon=":material/article:")
            st.caption(
                text(
                    "review_notice"
                    if literature_review["ai_generated"]
                    else "no_evidence_report_notice"
                )
            )
            with st.container(border=True):
                st.markdown(literature_review["content"])
            st.download_button(
                text("download_review_markdown"),
                data=literature_review["content"],
                file_name="botanical-literature-review.md",
                mime="text/markdown",
                key="download_botanical_review_markdown",
            )
            review_pdf = st.session_state.get("literature_review_pdf")
            if (
                review_pdf
                and review_pdf.get("question")
                == st.session_state["research_query"]
            ):
                st.download_button(
                    text("download_review_pdf"),
                    data=review_pdf["content"],
                    file_name="botanical-literature-review.pdf",
                    mime="application/pdf",
                    key="download_botanical_review_pdf",
                    icon=":material/download:",
                )
            elif st.button(
                text("retry_review_pdf"),
                key="retry_botanical_review_pdf",
            ):
                generate_review_pdf(
                    literature_review["content"],
                    st.session_state["research_query"],
                )

        if selected_language != "English":
            if st.button(
                f"{text('translate_report')} · {LANGUAGES[selected_language]}",
                type="primary",
                icon=":material/translate:",
                key="translate_research_report",
            ):
                try:
                    with st.spinner(text("translate_spinner")):
                        translated = translate_report(report_text, selected_language)
                    st.session_state["translated_report"] = {
                        "language": selected_language,
                        "content": translated,
                    }
                except Exception:
                    logger.exception(
                        "Research report translation failed for %s",
                        selected_language,
                    )
                    st.error(text("translation_failed"))

            translated_report = st.session_state.get("translated_report")
            if (
                translated_report
                and translated_report["language"] == selected_language
            ):
                st.subheader(text("translated_report"), icon=":material/translate:")
                st.caption(text("generated_translation_notice"))
                with st.container(border=True):
                    st.markdown(translated_report["content"])
                st.download_button(
                    text("download_report"),
                    data=translated_report["content"],
                    file_name=(
                        "botanical-research-report-"
                        f"{selected_language.lower().replace(' ', '-')}.md"
                    ),
                    mime="text/markdown",
                    icon=":material/download:",
                    key="download_translated_report",
                )
