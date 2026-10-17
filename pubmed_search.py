from literature_search import search_literature


def search_pubmed(search_term):
    """Compatibility wrapper for scripts that previously used PubMed only."""
    result = search_literature(search_term, max_results=10)
    return result["papers"]


# Ask the user what to search
if __name__ == "__main__":
    search_term = input("What should I search for? ")

    papers = search_pubmed(search_term)

    print()
    print("Papers found:", len(papers))

    for paper in papers:
        print()
        print("PMID:", paper["pmid"])
        print("Title:", paper["title"])
        print("Journal:", paper["journal"])
        print("Abstract:", paper["abstract"][:500])
