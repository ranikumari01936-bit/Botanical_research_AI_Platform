import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _client() -> OpenAI:
    load_dotenv()
    return OpenAI(timeout=60.0, max_retries=1)


def _model_name() -> str:
    return os.getenv("BOTANICAL_AI_MODEL", "gpt-4o-mini")


def extract_information(paper):
    prompt = f"""
You are a scientific literature extraction assistant.

Read the following retrieved research paper information. Only its title and
PubMed abstract are available; do not imply that you reviewed the full paper.

Title:
{paper["title"]}

Abstract:
{paper["abstract"]}

Extract the following information:

1. Crop or plant studied
2. Main research problem
3. Stress or condition studied
4. Treatments or interventions mentioned
5. Important biological or physiological findings
6. Important genes, proteins, or molecular markers mentioned
7. Important environmental conditions mentioned
8. Main conclusion

Only use information present in the paper information provided.
Do not invent missing information.
If a requested detail is absent from the title and abstract, explicitly say it
is not specified in the retrieved abstract.

Return the answer with clear headings.
"""

    response = _client().responses.create(
        model=_model_name(),
        input=prompt,
    )

    extracted = response.output_text.strip()
    if not extracted:
        raise ValueError("The AI provider returned an empty evidence extraction.")
    return extracted








if __name__ == "__main__":
    test_paper = {
        "title": "Drought stress effects on tomato plants",
        "abstract": """
        Drought is an important environmental stress affecting tomato growth
        and productivity. The study investigated physiological responses of
        tomato plants under drought conditions. Drought reduced biomass
        accumulation and affected fruit production.
        """
    }

    result = extract_information(test_paper)

    print("\nEXTRACTION RESULT:")
    print(result)