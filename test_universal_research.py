import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest

import literature_search
from flower_classifier import predict_flower
from protocol_generator import generate_literature_review
from report_pdf import generate_report_pdf


class FakeResponse:
    def __init__(self, payload=None, content=b"", url=""):
        self.payload = payload
        self.content = content
        self.url = url
        self.headers = {}
        self.is_redirect = False
        self.closed = False
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        if self.content:
            yield self.content

    def close(self):
        self.closed = True


def europe_pmc_payload(
    title="Neem leaf study",
    pmid="12345",
    pmcid="",
    abstract="The abstract reports plant findings.",
    full_text_urls=None,
):
    record = {
        "id": pmid or pmcid,
        "source": "MED" if pmid else "PMC",
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": "10.1000/example",
        "title": title,
        "authorString": "A. Botanist; B. Researcher",
        "firstPublicationDate": "2024-03-02",
        "journalTitle": "Botanical Journal",
        "abstractText": abstract,
        "isOpenAccess": "Y" if full_text_urls else "N",
        "fullTextUrlList": {"fullTextUrl": full_text_urls or []},
    }
    return {"resultList": {"result": [record]}}


class LiteratureSearchTests(unittest.TestCase):
    def setUp(self):
        literature_search.search_literature.clear()
        literature_search.resolve_taxon.clear()

    def tearDown(self):
        literature_search.search_literature.clear()
        literature_search.resolve_taxon.clear()

    def test_accepts_common_scientific_and_research_queries(self):
        examples = [
            "Rose",
            "Neem",
            "Mango",
            "Azadirachta indica",
            "Xyznotum botanicum",
            "cultivation of an unfamiliar alpine herb",
        ]
        for query in examples:
            with self.subTest(query=query):
                literature_search.search_literature.clear()
                payload = europe_pmc_payload(title=f"Study of {query}")
                with (
                    patch(
                        "literature_search._europe_pmc_records",
                        return_value=[
                            literature_search._normalize_europe_pmc_record(
                                payload["resultList"]["result"][0]
                            )
                        ],
                    ) as europe_search,
                    patch(
                        "literature_search._pubmed_records",
                        return_value=[],
                    ) as pubmed_search,
                ):
                    result = literature_search.search_literature(
                        query,
                        max_results=1,
                    )
                self.assertEqual(len(result["papers"]), 1)
                europe_search.assert_called_once_with(query, 1)
                pubmed_search.assert_called_once_with(query, 1)
                self.assertEqual(result["sources"], ["Europe PMC", "PubMed"])

    def test_expands_resolved_common_name_with_verified_scientific_name(self):
        with (
            patch(
                "literature_search._europe_pmc_records",
                return_value=[],
            ) as europe_search,
            patch("literature_search._pubmed_records", return_value=[]),
        ):
            literature_search.search_literature(
                "Neem",
                max_results=1,
                scientific_name="Azadirachta indica",
            )
        self.assertIn(
            "Azadirachta indica",
            europe_search.call_args.args[0],
        )

    def test_no_results_are_explicit_and_not_fabricated(self):
        responses = [
            FakeResponse({"resultList": {"result": []}}),
            FakeResponse({"esearchresult": {"idlist": []}}),
        ]
        with patch(
            "literature_search.requests.get",
            side_effect=responses,
        ):
            result = literature_search.search_literature(
                "Unfamiliar plant species",
                max_results=5,
            )
        self.assertEqual(result["papers"], [])
        self.assertTrue(result["search_date"])

    def test_all_source_failures_raise_an_explicit_search_error(self):
        with patch(
            "literature_search.requests.get",
            side_effect=literature_search.requests.ConnectionError("offline"),
        ):
            with self.assertRaises(literature_search.LiteratureSearchError):
                literature_search.search_literature("mango", max_results=5)

    def test_metadata_without_an_abstract_is_retained(self):
        payload = europe_pmc_payload(abstract="")
        with patch(
            "literature_search.requests.get",
            return_value=FakeResponse(payload),
        ):
            result = literature_search.search_literature("Neem", max_results=1)
        paper = result["papers"][0]
        self.assertEqual(paper["abstract"], "")
        self.assertEqual(paper["authors"], "A. Botanist; B. Researcher")
        self.assertEqual(paper["pub_year"], "2024")

    def test_free_pdf_links_are_limited_to_trusted_source_metadata(self):
        payload = europe_pmc_payload(
            pmid="",
            pmcid="PMC123",
            full_text_urls=[
                {
                    "availability": "Free",
                    "documentStyle": "pdf",
                    "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/pdf/",
                    "site": "PubMedCentral",
                },
                {
                    "availability": "Subscription",
                    "documentStyle": "pdf",
                    "url": "https://www.ncbi.nlm.nih.gov/paid.pdf",
                    "site": "Publisher",
                },
                {
                    "availability": "Free",
                    "documentStyle": "pdf",
                    "url": "https://example.com/fake.pdf",
                    "site": "Unknown",
                },
            ],
        )
        paper = literature_search._normalize_europe_pmc_record(
            payload["resultList"]["result"][0]
        )
        self.assertEqual(
            paper["pdf_urls"],
            ["https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/pdf/"],
        )
        self.assertTrue(paper["is_open_access"])

    def test_verified_original_pdf_is_downloaded_and_magic_checked(self):
        pdf = b"%PDF-1.7\noriginal paper\n%%EOF"
        response = FakeResponse(
            content=pdf,
            url="https://www.ebi.ac.uk/articles/original.pdf",
        )
        paper = {"pdf_urls": ["https://www.ebi.ac.uk/article.pdf"]}
        with patch("literature_search.requests.get", return_value=response):
            data, error = literature_search.download_original_pdf(paper)
        self.assertEqual(data, pdf)
        self.assertEqual(error, "")
        self.assertTrue(response.closed)

    def test_original_pdf_rejects_untrusted_redirects(self):
        response = FakeResponse()
        response.is_redirect = True
        response.headers = {"Location": "http://127.0.0.1/private"}
        with patch(
            "literature_search.requests.get",
            return_value=response,
        ) as get:
            data, error = literature_search.download_original_pdf({
                "pdf_urls": ["https://www.ebi.ac.uk/article.pdf"],
            })
        self.assertIsNone(data)
        self.assertTrue(error)
        get.assert_called_once()
        self.assertTrue(response.closed)

    def test_invalid_pdf_response_is_not_offered_as_original(self):
        response = FakeResponse(
            content=b"<html>login page</html>",
            url="https://www.ebi.ac.uk/article.pdf",
        )
        with patch("literature_search.requests.get", return_value=response):
            data, error = literature_search.download_original_pdf({
                "pdf_urls": ["https://www.ebi.ac.uk/article.pdf"],
            })
        self.assertIsNone(data)
        self.assertIn("valid PDF", error)
        self.assertTrue(response.closed)

    def test_scientific_name_resolves_to_verified_plant_record(self):
        match = {
            "scientificName": "Azadirachta indica A.Juss.",
            "canonicalName": "Azadirachta indica",
            "rank": "SPECIES",
            "kingdom": "Plantae",
            "confidence": 99,
            "matchType": "EXACT",
            "status": "ACCEPTED",
            "key": 123,
        }
        with patch(
            "literature_search.requests.get",
            return_value=FakeResponse(match),
        ):
            result = literature_search.resolve_taxon("Azadirachta indica")
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["scientific_name"], "Azadirachta indica")

    def test_ambiguous_common_name_requires_user_choice(self):
        def gbif_response(url, params=None, **kwargs):
            if url.endswith("/species/match"):
                return FakeResponse({"matchType": "NONE"})
            if url.endswith("/species/search"):
                return FakeResponse({
                    "results": [
                        {
                            "key": 11,
                            "scientificName": "Mangifera indica L.",
                            "canonicalName": "Mangifera indica",
                            "rank": "SPECIES",
                            "kingdom": "Plantae",
                        },
                        {
                            "key": 12,
                            "scientificName": "Mangifera caesia Jack",
                            "canonicalName": "Mangifera caesia",
                            "rank": "SPECIES",
                            "kingdom": "Plantae",
                        },
                    ],
                })
            key = url.split("/")[-2]
            names = {
                "11": [{"vernacularName": "Mango"}],
                "12": [{"vernacularName": "Mango"}],
            }
            return FakeResponse({"results": names[key]})

        with patch(
            "literature_search.requests.get",
            side_effect=gbif_response,
        ):
            result = literature_search.resolve_taxon("Mango")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(
            {item["canonical_name"] for item in result["candidates"]},
            {"Mangifera indica", "Mangifera caesia"},
        )

    def test_unfamiliar_name_is_not_assigned_a_guessed_taxon(self):
        def gbif_response(url, params=None, **kwargs):
            if url.endswith("/species/match"):
                return FakeResponse({"matchType": "NONE"})
            return FakeResponse({"results": []})

        with patch(
            "literature_search.requests.get",
            side_effect=gbif_response,
        ):
            result = literature_search.resolve_taxon(
                "Xyznotum botanicum"
            )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["scientific_name"], "")

    def test_duplicate_pubmed_and_pmc_records_merge_by_shared_ids(self):
        e_pmc = _normalize = literature_search._normalize_europe_pmc_record(
            europe_pmc_payload(
                pmid="",
                pmcid="PMC123",
                title="Neem leaf study",
            )["resultList"]["result"][0]
        )
        pubmed_xml = """<?xml version="1.0"?>
        <PubmedArticleSet><PubmedArticle>
          <MedlineCitation><PMID>12345</PMID><Article>
            <ArticleTitle>Neem leaf study</ArticleTitle>
            <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
              <Title>Botanical Journal</Title></Journal>
            <Abstract><AbstractText>Plant evidence.</AbstractText></Abstract>
          </Article></MedlineCitation>
          <PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC123</ArticleId>
            <ArticleId IdType="doi">10.1000/example</ArticleId></ArticleIdList></PubmedData>
        </PubmedArticle></PubmedArticleSet>"""
        responses = [
            FakeResponse({"esearchresult": {"idlist": ["12345"]}}),
            FakeResponse(content=pubmed_xml.encode("utf-8")),
        ]
        with patch(
            "literature_search.requests.get",
            side_effect=[
                FakeResponse({
                    "resultList": {"result": [{
                        "id": "PMC123",
                        "source": "PMC",
                        "pmcid": "PMC123",
                        "title": "Neem leaf study",
                        "fullTextUrlList": {},
                    }]},
                }),
                *responses,
            ],
        ):
            result = literature_search.search_literature("Neem", max_results=2)
        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["pmid"], "12345")
        self.assertEqual(result["papers"][0]["abstract"], "Plant evidence.")

    def test_gbif_synonyms_are_returned_for_search_expansion(self):
        response = FakeResponse({
            "results": [
                {"canonicalName": "Melia azadirachta"},
                {"scientificName": "Azadirachta indica A.Juss."},
                {"canonicalName": "Melia azadirachta"},
            ]
        })
        with patch("literature_search.requests.get", return_value=response) as get:
            synonyms = literature_search.get_taxon_synonyms("123")
        self.assertEqual(
            synonyms,
            ["Melia azadirachta", "Azadirachta indica A.Juss."],
        )
        self.assertIn("/species/123/synonyms", get.call_args.args[0])


class WorkflowTests(unittest.TestCase):
    def test_papers_without_abstracts_skip_ai_extraction_with_notice(self):
        from workflow import extraction_node

        with patch("workflow.extract_information") as extract:
            result = extraction_node({
                "question": "Neem",
                "papers": [{
                    "pmid": "123",
                    "title": "Study without an abstract",
                    "abstract": "",
                }],
            })
        extract.assert_not_called()
        self.assertEqual(result["extracted_information"], [])
        self.assertIn("No abstract", result["extraction_errors"][0]["error"])


class ReportGenerationTests(unittest.TestCase):
    def test_ai_review_only_keeps_supplied_citations_and_adds_verified_refs(self):
        response = SimpleNamespace(
            output_text=(
                "# Review\nEvidence [PMID: 12345].\n\n"
                "## References\nInvented reference."
            )
        )
        create = Mock(return_value=response)
        client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        paper = {
            "pmid": "12345",
            "title": "Neem study",
            "authors": "A. Botanist",
            "pub_year": "2024",
            "journal": "Plant Journal",
            "abstract": "The study reports a finding.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        }
        with patch("protocol_generator._client", return_value=client):
            report = generate_literature_review(
                "Neem",
                [paper],
                [],
                "2026-10-17",
            )
        self.assertIn("[PMID: 12345]", report)
        self.assertNotIn("99999", report)
        self.assertNotIn("Invented reference", report)
        self.assertIn("Verified source record", report)
        self.assertIn("abstract-based", create.call_args.kwargs["input"])

    def test_no_results_report_does_not_claim_findings(self):
        report = generate_literature_review(
            "unfamiliar plant",
            [],
            [],
            "2026-10-17",
        )
        self.assertIn("No literature records", report)
        self.assertIn("does not prove", report)
        self.assertIn("No records retrieved", report)

    def test_review_rejects_citations_not_present_in_retrieved_records(self):
        response = SimpleNamespace(
            output_text="An unsupported claim [PMID: 99999]."
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: response,
            )
        )
        with patch("protocol_generator._client", return_value=client):
            with self.assertRaisesRegex(ValueError, "identifiers"):
                generate_literature_review(
                    "Neem",
                    [{"pmid": "12345", "title": "Verified paper"}],
                    [],
                    "2026-10-17",
                )

    def test_report_pdf_spans_multiple_pages_for_long_reports(self):
        report = "\n\n".join(
            f"## Evidence section {index}\n\n"
            + ("Retrieved abstract evidence with a verified citation. " * 70)
            for index in range(18)
        )
        pdf = generate_report_pdf(report)
        self.assertGreaterEqual(pdf.count(b"/Type /Page\n"), 2)

    def test_report_pdf_handles_long_text_tables_and_references(self):
        long_report = "\n\n".join([
            "# Botanical literature review",
            "## Abstract",
            "Plant evidence. " * 400,
            "## Comparison",
            "| Study | Finding |",
            "| --- | --- |",
            *[
                f"| Study {index} | {'evidence ' * 30} |"
                for index in range(80)
            ],
            "## References",
            *[
                f"- Author {index}. Long plant research reference and source "
                "details. https://pubmed.ncbi.nlm.nih.gov/12345/"
                for index in range(30)
            ],
        ])
        pdf = generate_report_pdf(long_report)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 5000)

    def test_empty_report_pdf_fails_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "report is empty"):
            generate_report_pdf("")


class ExistingClassifierTests(unittest.TestCase):
    def test_existing_classifier_stays_within_its_five_classes(self):
        project = Path(__file__).resolve().parent
        classes = json.loads(
            (project / "plant_ml" / "models" / "flower_classes.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            classes,
            ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
        )
        with Image.new("RGB", (320, 240), color=(71, 123, 85)) as image:
            result = predict_flower(image.copy())
        self.assertIn(result["flower"], classes)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)


class ResearchWorkspaceTests(unittest.TestCase):
    def test_research_workspace_is_available_without_an_image_upload(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parent / "app.py"),
            default_timeout=60,
        )
        app.session_state["workspace"] = "Research workspace"
        app.session_state["research_topic"] = "Mangifera indica"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.file_uploader), 0)
        self.assertEqual(app.text_input[0].value, "Mangifera indica")
        self.assertTrue(any(
            widget.label == "Maximum candidate papers to retrieve"
            for widget in app.select_slider
        ))


if __name__ == "__main__":
    unittest.main()
