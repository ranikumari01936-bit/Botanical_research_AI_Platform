import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from literature_search import search_literature
from extraction_agent import extract_information
from protocol_generator import generate_research_plan


logger = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    question: str
    max_results: int
    scientific_name: str
    taxonomy: dict[str, Any]
    papers: list[dict[str, Any]]
    extracted_information: list[dict[str, Any]]
    extraction_errors: list[dict[str, str]]
    search_warnings: list[str]
    search_date: str
    search_sources: list[str]
    searched_taxon_terms: list[str]
    plan_error: str
    final_plan: str


def search_node(state: ResearchState):
    search_result = search_literature(
        state["question"],
        max_results=state.get("max_results", 10),
        scientific_name=state.get("scientific_name"),
        synonyms=state.get("taxonomy", {}).get("synonyms", []),
    )
    search_warnings = list(search_result["warnings"])
    synonym_warning = state.get("taxonomy", {}).get("synonym_warning")
    if synonym_warning:
        search_warnings.append(synonym_warning)
    return {
        "question": state["question"],
        "papers": search_result["papers"],
        "search_warnings": search_warnings,
        "search_date": search_result["search_date"],
        "search_sources": search_result["sources"],
        "searched_taxon_terms": search_result["searched_taxon_terms"],
        "taxonomy": state.get("taxonomy", {}),
        "extraction_errors": [],
    }


def extraction_node(state: ResearchState):
    extracted_information = []
    extraction_errors: list[dict[str, str]] = []

    for paper in state.get("papers", []):
        title = str(paper.get("title") or "Untitled paper")
        if not paper.get("abstract"):
            extraction_errors.append({
                "title": title,
                "error": "No abstract was retrieved; evidence extraction was skipped.",
            })
            continue
        try:
            extracted = extract_information(paper)
            extracted_information.append({
                "pmid": paper.get("pmid", ""),
                "pmcid": paper.get("pmcid", ""),
                "doi": paper.get("doi", ""),
                "title": title,
                "extracted": extracted,
            })
        except Exception as error:
            logger.exception("Evidence extraction failed for %s", title)
            extraction_errors.append({
                "title": title,
                "error": str(error),
            })

    return {
        "question": state["question"],
        "papers": state.get("papers", []),
        "extracted_information": extracted_information,
        "extraction_errors": extraction_errors,
    }


def protocol_node(state: ResearchState):
    combined_information = ""
    papers_by_id = {
        str(paper.get("pmid") or paper.get("pmcid") or paper.get("doi") or ""): paper
        for paper in state["papers"]
    }

    for item in state["extracted_information"]:
        identifier = str(
            item.get("pmid") or item.get("pmcid") or item.get("doi") or ""
        )
        paper = papers_by_id.get(identifier, {})

        combined_information += "\n\n"
        combined_information += "PAPER TITLE:\n"
        combined_information += str(item.get("title") or "Untitled paper")

        combined_information += "\nSOURCE IDENTIFIER:\n"
        combined_information += identifier or "Not available"

        combined_information += "\nRETRIEVED ABSTRACT:\n"
        combined_information += str(
            paper.get("abstract") or "No abstract was retrieved."
        )

        combined_information += "\nAI EXTRACTED INFORMATION:\n"
        combined_information += str(item.get("extracted") or "")

    final_plan = ""
    plan_error = ""
    if state.get("papers") and state.get("extracted_information"):
        try:
            final_plan = generate_research_plan(combined_information)
        except Exception as error:
            logger.exception("Evidence-based research plan generation failed")
            plan_error = str(error)
    return {
        "question": state["question"],
        "papers": state.get("papers", []),
        "extracted_information": state.get("extracted_information", []),
        "extraction_errors": state.get("extraction_errors", []),
        "search_warnings": state.get("search_warnings", []),
        "search_date": state.get("search_date", ""),
        "search_sources": state.get("search_sources", []),
        "searched_taxon_terms": state.get("searched_taxon_terms", []),
        "taxonomy": state.get("taxonomy", {}),
        "final_plan": final_plan,
        "plan_error": plan_error,
    }


graph = StateGraph(ResearchState)

graph.add_node("search", search_node)
graph.add_node("extract", extraction_node)
graph.add_node("protocol", protocol_node)

graph.add_edge(START, "search")
graph.add_edge("search", "extract")
graph.add_edge("extract", "protocol")
graph.add_edge("protocol", END)

app = graph.compile()


if __name__ == "__main__":
    question = input("Research question: ")


    result = app.invoke({
        "question": question
    })

    print("\nWORKFLOW FINISHED")

    print("Question:", result["question"])

    print(
        "Number of papers:",
        len(result["papers"])
    )

    print(
        "Number of extracted papers:",
        len(result["extracted_information"])
    )

    print("\nFINAL RESEARCH PLAN")
    print("=" * 60)

    print(result["final_plan"])