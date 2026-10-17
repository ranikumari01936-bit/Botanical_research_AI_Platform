import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any
from urllib.parse import quote, urljoin, urlparse
import xml.etree.ElementTree as ET

import requests
import streamlit as st


logger = logging.getLogger(__name__)
EUROPE_PMC_SEARCH_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)
PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)
GBIF_API_URL = "https://api.gbif.org/v1"
REQUEST_TIMEOUT = (5, 20)
MAX_PAPERS = 30
MAX_PDF_BYTES = 40 * 1024 * 1024
_STOP_WORDS = {
    "about", "and", "are", "based", "by", "for", "from", "how", "in",
    "is", "of", "on", "or", "the", "to", "what", "which", "with",
}
_RESEARCH_TERMS = {
    "cultivation", "disease", "drought", "effect", "effects", "growth",
    "growing", "genetic", "genetics", "literature", "medicinal",
    "research", "response", "soil", "stress", "temperature", "yield",
}
_OFFICIAL_SOURCE_HOSTS = {
    "europepmc.org",
    "www.ebi.ac.uk",
    "www.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "cdn.ncbi.nlm.nih.gov",
}


class LiteratureSearchError(RuntimeError):
    """Raised when all configured scientific literature sources fail."""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _is_official_https_url(value: str) -> bool:
    parsed = urlparse(value)
    try:
        return (
            parsed.scheme == "https"
            and parsed.hostname in _OFFICIAL_SOURCE_HOSTS
            and parsed.port is None
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def _paper_source_url(paper: dict[str, Any]) -> str:
    pmcid = str(paper.get("pmcid") or "")
    if re.fullmatch(r"PMC\d+", pmcid):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

    pmid = str(paper.get("pmid") or "")
    if pmid.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    doi = str(paper.get("doi") or "")
    if re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        return f"https://doi.org/{quote(doi, safe='/')}"
    return ""


def _normalize_europe_pmc_record(record: dict[str, Any]) -> dict[str, Any]:
    pmcid = str(record.get("pmcid") or "")
    if not pmcid and str(record.get("source", "")).upper() == "PMC":
        record_id = str(record.get("id") or "")
        pmcid = record_id if re.fullmatch(r"PMC\d+", record_id) else ""

    journal_info = record.get("journalInfo") or {}
    journal = record.get("journalTitle")
    if not journal and isinstance(journal_info, dict):
        journal = (journal_info.get("journal") or {}).get("title")

    full_text_urls: list[dict[str, str]] = []
    pdf_urls: list[str] = []
    url_data = record.get("fullTextUrlList") or {}
    entries = url_data.get("fullTextUrl", []) if isinstance(url_data, dict) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "")
        if not _is_official_https_url(url):
            continue
        style = str(entry.get("documentStyle") or "").lower()
        availability = str(entry.get("availability") or "").lower()
        full_text_urls.append({
            "url": url,
            "style": style,
            "availability": availability,
            "site": str(entry.get("site") or ""),
        })
        if style == "pdf" and availability in {"free", "open access"}:
            pdf_urls.append(url)

    paper = {
        "pmid": str(record.get("pmid") or ""),
        "pmcid": pmcid,
        "doi": str(record.get("doi") or ""),
        "title": _clean_text(record.get("title")) or "Title not provided",
        "authors": _clean_text(record.get("authorString")),
        "pub_year": (
            str(record.get("firstPublicationDate") or "")[:4]
            or str(record.get("pubYear") or "")
        ),
        "journal": _clean_text(journal),
        "abstract": _clean_text(record.get("abstractText")),
        "source": "Europe PMC",
        "is_open_access": str(record.get("isOpenAccess", "")).upper() == "Y",
        "full_text_urls": full_text_urls,
        "pdf_urls": list(dict.fromkeys(pdf_urls)),
    }
    paper["full_text_available"] = bool(full_text_urls or paper["pmcid"])
    paper["source_url"] = _paper_source_url(paper)
    return paper


def _pubmed_records(query: str, max_results: int) -> list[dict[str, Any]]:
    search_response = requests.get(
        PUBMED_SEARCH_URL,
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        },
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
    )
    search_response.raise_for_status()
    ids = search_response.json()["esearchresult"]["idlist"]
    if not ids:
        return []

    fetch_response = requests.get(
        PUBMED_FETCH_URL,
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
    )
    fetch_response.raise_for_status()
    root = ET.fromstring(fetch_response.text)
    papers: list[dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        citation = article.find("./MedlineCitation")
        article_data = citation.find("./Article") if citation is not None else None
        if article_data is None:
            continue

        title_element = article_data.find("./ArticleTitle")
        title = (
            "".join(title_element.itertext()).strip()
            if title_element is not None
            else "Title not provided"
        )
        abstract_parts = [
            _clean_text("".join(element.itertext()))
            for element in article_data.findall("./Abstract/AbstractText")
        ]
        authors = []
        for author in article_data.findall("./AuthorList/Author"):
            collective_name = author.findtext("./CollectiveName")
            if collective_name:
                authors.append(collective_name)
                continue
            name = " ".join(filter(None, (
                author.findtext("./ForeName"),
                author.findtext("./LastName"),
            )))
            if name:
                authors.append(name)

        article_ids = {
            item.attrib.get("IdType", "").lower(): _clean_text("".join(item.itertext()))
            for item in article.findall("./PubmedData/ArticleIdList/ArticleId")
        }
        journal = article_data.findtext("./Journal/Title")
        pub_date = article_data.find("./Journal/JournalIssue/PubDate")
        year = ""
        if pub_date is not None:
            year = pub_date.findtext("./Year") or ""
            if not year:
                medline_date = pub_date.findtext("./MedlineDate") or ""
                year_match = re.search(r"\b(?:19|20)\d{2}\b", medline_date)
                year = year_match.group(0) if year_match else ""

        pmid = citation.findtext("./PMID") if citation is not None else ""
        paper = {
            "pmid": str(pmid or ""),
            "pmcid": article_ids.get("pmc", ""),
            "doi": article_ids.get("doi", ""),
            "title": _clean_text(title) or "Title not provided",
            "authors": "; ".join(authors),
            "pub_year": year,
            "journal": _clean_text(journal),
            "abstract": " ".join(part for part in abstract_parts if part),
            "source": "PubMed",
            "is_open_access": False,
            "full_text_available": bool(article_ids.get("pmc")),
            "full_text_urls": [],
            "pdf_urls": [],
        }
        if paper["pmcid"]:
            paper["full_text_urls"].append({
                "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{paper['pmcid']}/",
                "style": "html",
                "availability": "Unverified",
                "site": "PubMed Central",
            })
        paper["source_url"] = _paper_source_url(paper)
        papers.append(paper)
    return papers


def _europe_pmc_records(
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    response = requests.get(
        EUROPE_PMC_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core",
        },
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
    )
    response.raise_for_status()
    records = response.json().get("resultList", {}).get("result", [])
    return [
        _normalize_europe_pmc_record(record)
        for record in records
        if isinstance(record, dict)
    ]


def _paper_keys(paper: dict[str, Any]) -> list[str]:
    keys = [
        f"{key}:{value.strip().lower()}"
        for key in ("pmid", "pmcid", "doi")
        if (value := str(paper.get(key) or "").strip())
    ]
    if not keys:
        keys.append(
            "title:"
            + re.sub(r"\W+", "", str(paper.get("title", "")).lower())
        )
    return keys


def _merge_papers(
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    aliases: dict[str, int] = {}
    for paper in papers:
        keys = _paper_keys(paper)
        existing_index = next(
            (aliases[key] for key in keys if key in aliases),
            None,
        )
        if existing_index is None:
            existing_index = len(merged)
            merged.append(paper.copy())
        else:
            target = merged[existing_index]
            for field in (
                "pmid", "pmcid", "doi", "authors", "pub_year", "journal",
                "abstract", "source_url",
            ):
                if not target.get(field) and paper.get(field):
                    target[field] = paper[field]
            target["is_open_access"] = (
                target.get("is_open_access") or paper.get("is_open_access")
            )
            target["full_text_available"] = (
                target.get("full_text_available")
                or paper.get("full_text_available")
            )
            for field in ("full_text_urls", "pdf_urls"):
                combined = target.get(field, []) + paper.get(field, [])
                if field == "full_text_urls":
                    unique_urls = {}
                    for item in combined:
                        if isinstance(item, dict) and item.get("url"):
                            key = (
                                item["url"],
                                item.get("style", ""),
                                item.get("availability", ""),
                            )
                            unique_urls.setdefault(key, item)
                    target[field] = list(unique_urls.values())
                else:
                    target[field] = list(dict.fromkeys(combined))
            if not target.get("source") and paper.get("source"):
                target["source"] = paper["source"]
            target["source_url"] = (
                _paper_source_url(target) or target.get("source_url", "")
            )
        for key in _paper_keys(merged[existing_index]):
            aliases[key] = existing_index
        for key in keys:
            aliases[key] = existing_index
    return merged


def _relevance_score(
    paper: dict[str, Any],
    terms: set[str],
    phrase: str,
) -> float:
    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    title_terms = set(re.findall(r"[a-z0-9]+", title))
    abstract_terms = set(re.findall(r"[a-z0-9]+", abstract))
    score = 4 * len(terms & title_terms) + len(terms & abstract_terms)
    normalized_title = re.sub(r"\s+", " ", title)
    if phrase and phrase in normalized_title:
        score += 8
    return float(score)


@st.cache_data(ttl="12h", max_entries=64, show_spinner=False)
def search_literature(
    query: str,
    max_results: int = 10,
    scientific_name: str | None = None,
    synonyms: list[str] | None = None,
) -> dict[str, Any]:
    """Search Europe PMC and PubMed, merging duplicate verified records."""
    clean_query = " ".join(query.split()).strip()
    if not clean_query:
        raise ValueError("Enter a plant name or botanical research question.")
    if len(clean_query) > 500:
        raise ValueError("Research questions must be 500 characters or fewer.")
    if not 1 <= max_results <= MAX_PAPERS:
        raise ValueError(f"Choose between 1 and {MAX_PAPERS} papers.")

    scientific_terms = list(dict.fromkeys(
        name.strip()
        for name in [scientific_name, *(synonyms or [])]
        if name and name.strip()
    ))
    api_query = clean_query
    for name in scientific_terms:
        if name.casefold() not in api_query.casefold():
            safe_name = name.replace('"', "")
            api_query += f' OR ("{safe_name}")'

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    source_failures: list[str] = []
    source_searches = (
        ("Europe PMC", _europe_pmc_records),
        ("PubMed", _pubmed_records),
    )

    with ThreadPoolExecutor(max_workers=len(source_searches)) as executor:
        pending_searches = [
            (source, executor.submit(search, api_query, max_results))
            for source, search in source_searches
        ]
        for source, pending in pending_searches:
            try:
                records.extend(pending.result())
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                ET.ParseError,
            ) as error:
                logger.warning("%s search failed: %s", source, error)
                source_failures.append(source)

    if source_failures:
        if not records and len(source_failures) == 2:
            raise LiteratureSearchError(
                "Europe PMC and PubMed could not be reached. "
                "Check your connection and try again."
            )
        warnings.append(
            "Some literature sources could not be reached: "
            + ", ".join(source_failures)
            + ". Results may be incomplete."
        )

    terms = {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            f"{clean_query} {' '.join(scientific_terms)}".lower(),
        )
        if len(token) > 1 and token not in _STOP_WORDS
    }
    phrase = clean_query.casefold() if len(clean_query.split()) <= 3 else ""
    unique_papers = _merge_papers(records)
    unique_papers.sort(
        key=lambda paper: _relevance_score(paper, terms, phrase),
        reverse=True,
    )
    papers = unique_papers[:max_results]
    return {
        "papers": papers,
        "warnings": warnings,
        "search_date": date.today().isoformat(),
        "query": clean_query,
        "resolved_scientific_name": scientific_name or "",
        "searched_taxon_terms": scientific_terms,
        "sources": [source for source, _ in source_searches],
    }


def _is_name_only_query(query: str) -> bool:
    normalized = " ".join(query.strip().split())
    if not normalized or len(normalized) > 100:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*", normalized)
    if not 1 <= len(words) <= 3:
        return False
    return not any(word.casefold() in _RESEARCH_TERMS for word in words)


def _gbif_candidate(candidate: dict[str, Any]) -> dict[str, str]:
    key = str(candidate.get("key") or "")
    return {
        "scientific_name": _clean_text(candidate.get("scientificName")),
        "canonical_name": _clean_text(candidate.get("canonicalName")),
        "rank": _clean_text(candidate.get("rank")),
        "gbif_key": key,
        "source_url": f"https://www.gbif.org/species/{key}" if key.isdigit() else "",
    }


@st.cache_data(ttl="30d", max_entries=128, show_spinner=False)
def get_taxon_synonyms(gbif_key: str) -> list[str]:
    """Return GBIF-reported scientific synonyms for an accepted taxon."""
    if not str(gbif_key).isdigit():
        raise ValueError("A verified GBIF taxon key is required.")
    response = requests.get(
        f"{GBIF_API_URL}/species/{gbif_key}/synonyms",
        params={"limit": 100},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    names = []
    for item in results:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("canonicalName") or item.get("scientificName"))
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def _exact_common_name_candidates(query: str) -> list[dict[str, str]]:
    response = requests.get(
        f"{GBIF_API_URL}/species/search",
        params={
            "q": query,
            "highertaxon_key": 6,
            "status": "ACCEPTED",
            "limit": 20,
        },
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    plants = [
        _gbif_candidate(item)
        for item in results
        if isinstance(item, dict)
        and str(item.get("kingdom") or "").casefold() == "plantae"
        and item.get("key")
    ]
    if not plants:
        return []

    def has_exact_vernacular(candidate: dict[str, str]) -> bool:
        response = requests.get(
            f"{GBIF_API_URL}/species/{candidate['gbif_key']}/vernacularNames",
            params={"limit": 100},
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
        )
        response.raise_for_status()
        names = response.json().get("results", [])
        return any(
            _clean_text(item.get("vernacularName")).casefold() == query.casefold()
            for item in names
            if isinstance(item, dict)
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        matched = [
            candidate
            for candidate, is_match in zip(
                plants,
                executor.map(has_exact_vernacular, plants),
            )
            if is_match
        ]
    return matched


@st.cache_data(ttl="30d", max_entries=128, show_spinner=False)
def resolve_taxon(query: str) -> dict[str, Any]:
    """Resolve simple plant labels against GBIF; never infer from fuzzy hits."""
    clean_query = " ".join(query.split()).strip()
    if not _is_name_only_query(clean_query):
        return {
            "status": "not_applicable",
            "query": clean_query,
            "scientific_name": "",
            "candidates": [],
            "source": "",
        }

    try:
        match_response = requests.get(
            f"{GBIF_API_URL}/species/match",
            params={"name": clean_query, "strict": False},
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
        )
        match_response.raise_for_status()
        match = match_response.json()
    except (requests.RequestException, ValueError) as error:
        logger.warning("GBIF taxon match failed: %s", error)
        return {
            "status": "unavailable",
            "query": clean_query,
            "scientific_name": "",
            "candidates": [],
            "source": "",
        }

    if (
        str(match.get("kingdom") or "").casefold() == "plantae"
        and match.get("matchType") == "EXACT"
        and int(match.get("confidence") or 0) >= 90
        and match.get("canonicalName")
    ):
        candidate = _gbif_candidate(match)
        candidate["confidence"] = str(match.get("confidence"))
        return {
            "status": "resolved",
            "query": clean_query,
            "scientific_name": candidate["canonical_name"],
            "candidates": [candidate],
            "source": "GBIF",
        }

    try:
        candidates = _exact_common_name_candidates(clean_query)
    except (requests.RequestException, ValueError) as error:
        logger.warning("GBIF common-name search failed: %s", error)
        return {
            "status": "unavailable",
            "query": clean_query,
            "scientific_name": "",
            "candidates": [],
            "source": "",
        }

    if len(candidates) == 1:
        return {
            "status": "resolved",
            "query": clean_query,
            "scientific_name": candidates[0]["canonical_name"],
            "candidates": candidates,
            "source": "GBIF exact vernacular-name match",
        }
    if candidates:
        return {
            "status": "ambiguous",
            "query": clean_query,
            "scientific_name": "",
            "candidates": candidates,
            "source": "GBIF exact vernacular-name matches",
        }
    return {
        "status": "unresolved",
        "query": clean_query,
        "scientific_name": "",
        "candidates": [],
        "source": "",
    }


def download_original_pdf(paper: dict[str, Any]) -> tuple[bytes | None, str]:
    """Download an API-listed free PDF and verify the PDF signature."""
    candidates = paper.get("pdf_urls", [])
    if not candidates:
        return None, "No verified free PDF link was provided by the source."

    for url in candidates:
        if not _is_official_https_url(str(url)):
            continue
        current_url = str(url)
        for _ in range(6):
            if not _is_official_https_url(current_url):
                logger.warning("Rejected non-official PDF URL: %s", current_url)
                break
            response = None
            try:
                response = requests.get(
                    current_url,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "BotanicalResearchAssistant/1.0"},
                    stream=True,
                    allow_redirects=False,
                )
                if response.is_redirect:
                    redirect_url = urljoin(current_url, response.headers.get("Location", ""))
                    response.close()
                    if not _is_official_https_url(redirect_url):
                        logger.warning(
                            "Rejected PDF redirect outside verified sources: %s",
                            redirect_url,
                        )
                        break
                    current_url = redirect_url
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                total_size = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total_size += len(chunk)
                    if total_size > MAX_PDF_BYTES:
                        return None, "The source PDF exceeds the 40 MB download limit."
                    chunks.append(chunk)
                content = b"".join(chunks)
                if (
                    content.startswith(b"%PDF-")
                    and b"%%EOF" in content[-2048:]
                ):
                    return content, ""
                break
            except requests.RequestException as error:
                logger.warning("Original PDF download failed: %s", error)
                break
            finally:
                if response is not None:
                    response.close()
        else:
            logger.warning("Original PDF redirect limit exceeded: %s", url)

    return None, "The source did not return a valid PDF. Open the article page instead."
