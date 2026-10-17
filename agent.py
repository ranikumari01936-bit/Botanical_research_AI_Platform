from pubmed_search import search_pubmed
from extraction_agent import extract_information


def research_agent():

    search_term = input("What do you want to research? ")

    papers = search_pubmed(search_term)

    print()
    print("Found", len(papers), "papers.")

    for i, paper in enumerate(papers, start=1):

        print()
        print("=" * 60)
        print("PAPER", i)
        print("=" * 60)

        print("PMID:", paper["pmid"])
        print("Title:", paper["title"])

        print()
        print("Extracting information...")

        extracted = extract_information(paper)

        print()
        print("EXTRACTED INFORMATION:")
        print(extracted)


if __name__ == "__main__":
    research_agent()