"""Agent tools: drug interaction check (RxNorm) and PubMed search (NCBI E-utilities)."""

from __future__ import annotations

import re

import requests

from src.config import NCBI_API_KEY
from src.generation import generate_answer
from src.retrieval import hybrid_search, rerank, semantic_search

# --- Drug interaction (RxNorm / NLM) ---------------------------------------------------

COMMON_DRUGS = [
    "metformin", "lisinopril", "amlodipine", "atorvastatin", "metoprolol",
    "omeprazole", "losartan", "albuterol", "gabapentin", "hydrochlorothiazide",
    "sertraline", "acetaminophen", "ibuprofen", "amoxicillin", "azithromycin",
    "prednisone", "insulin", "warfarin", "clopidogrel", "furosemide",
    "carvedilol", "spironolactone", "empagliflozin", "dapagliflozin",
    "candesartan", "valsartan", "enalapril", "ramipril", "atenolol",
    "doxycycline", "ceftriaxone", "penicillin", "ciprofloxacin", "aspirin",
    "simvastatin", "rosuvastatin", "pantoprazole", "levothyroxine",
]


def check_drug_interactions(drug_names: list[str], timeout: float = 10.0) -> dict:
    """
    Look up drug interactions using the NLM RxNav interaction API.
    Returns a dict summarizing interactions found between the provided drugs.
    """
    interactions: list[dict] = []
    drugs_lower = [d.lower() for d in drug_names]

    for drug in drug_names:
        try:
            rxnorm_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug}&search=1"
            rxnorm_resp = requests.get(rxnorm_url, timeout=timeout)
            if rxnorm_resp.status_code != 200:
                continue

            rxcuis = rxnorm_resp.json().get("idGroup", {}).get("rxnormId", []) or []
            if not rxcuis:
                continue

            rxcui = rxcuis[0]
            int_url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}"
            int_resp = requests.get(int_url, timeout=timeout)
            if int_resp.status_code != 200:
                continue

            groups = int_resp.json().get("interactionTypeGroup", []) or []
            for group in groups:
                for itype in group.get("interactionType", []) or []:
                    for pair in itype.get("interactionPair", []) or []:
                        desc = pair.get("description", "") or ""
                        # Only surface an interaction pair if it references one of the
                        # OTHER drugs in the request list.
                        if any(
                            other in desc.lower()
                            for other in drugs_lower
                            if other != drug.lower()
                        ):
                            interactions.append(
                                {
                                    "drugs": drug_names,
                                    "description": desc,
                                    "severity": pair.get("severity", "Unknown"),
                                }
                            )
        except requests.RequestException as exc:
            print(f"  RxNorm lookup failed for '{drug}': {exc}")
            continue

    # Dedupe by description
    seen = set()
    unique: list[dict] = []
    for item in interactions:
        if item["description"] in seen:
            continue
        seen.add(item["description"])
        unique.append(item)

    return {
        "drugs_checked": drug_names,
        "interactions_found": len(unique),
        "interactions": unique,
    }


# --- PubMed (NCBI E-utilities) ---------------------------------------------------------


def search_pubmed(query: str, max_results: int = 3, timeout: float = 10.0) -> list[dict]:
    """Search PubMed for recent articles matching the query."""
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "date",
            "retmode": "json",
        }
        if NCBI_API_KEY:
            search_params["api_key"] = NCBI_API_KEY

        search_resp = requests.get(f"{base}/esearch.fcgi", params=search_params, timeout=timeout)
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", []) or []
        if not ids:
            return []

        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        if NCBI_API_KEY:
            fetch_params["api_key"] = NCBI_API_KEY
        fetch_resp = requests.get(f"{base}/esummary.fcgi", params=fetch_params, timeout=timeout)
        fetch_resp.raise_for_status()
        result = fetch_resp.json().get("result", {})
    except requests.RequestException as exc:
        print(f"  PubMed search failed: {exc}")
        return []

    articles: list[dict] = []
    for pmid in ids:
        article = result.get(pmid, {}) or {}
        if not article:
            continue
        authors = ", ".join(
            (a.get("name", "") for a in (article.get("authors") or [])[:3])
        )
        articles.append(
            {
                "pmid": pmid,
                "title": article.get("title", ""),
                "authors": authors,
                "journal": article.get("fulljournalname", ""),
                "pub_date": article.get("pubdate", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return articles


# --- Intent detection + agent orchestration --------------------------------------------


def detect_drugs_in_query(query: str) -> list[str]:
    """Detect medication names in a query."""
    query_lower = query.lower()
    # Use word boundaries to avoid substring hits (e.g., "insulinoma" vs "insulin")
    return [drug for drug in COMMON_DRUGS if re.search(rf"\b{re.escape(drug)}\b", query_lower)]


def detect_intent(query: str) -> dict:
    """Decide which tools should run for a given query."""
    query_lower = query.lower()

    intent = {
        "needs_rag": True,
        "needs_drug_check": False,
        "needs_pubmed": False,
        "detected_drugs": [],
    }

    drugs = detect_drugs_in_query(query)
    if len(drugs) >= 2 or "interaction" in query_lower:
        intent["needs_drug_check"] = True
        intent["detected_drugs"] = drugs

    research_keywords = [
        "recent", "latest", "new study", "new research", "evidence",
        "clinical trial", "trial", "2024", "2025", "2026",
    ]
    if any(keyword in query_lower for keyword in research_keywords):
        intent["needs_pubmed"] = True

    return intent


def run_agent(
    query: str,
    all_chunks: list[dict] | None,
    prompt_version: str = "v3_few_shot",
    model: str = "gpt-4o-mini",
    filter_metadata: dict | None = None,
    top_k: int = 5,
) -> dict:
    """Full agent pipeline: detect intent, run RAG + optional tools, generate answer."""
    intent = detect_intent(query)
    tools_used = ["rag"]

    # RAG retrieval - prefer hybrid when a chunk corpus is provided; otherwise pure semantic
    if all_chunks:
        retrieved = hybrid_search(query, all_chunks, top_k=top_k * 2, filter_metadata=filter_metadata)
    else:
        retrieved = semantic_search(query, top_k=top_k * 2, filter_metadata=filter_metadata)

    retrieved = rerank(query, retrieved, top_k=top_k)

    drug_info = None
    if intent["needs_drug_check"] and intent["detected_drugs"]:
        drug_info = check_drug_interactions(intent["detected_drugs"])
        tools_used.append("drug_interaction_check")

    pubmed_results = None
    if intent["needs_pubmed"]:
        pubmed_results = search_pubmed(query, max_results=3)
        tools_used.append("pubmed_search")

    # Augment context with tool results so the LLM can cite them in the answer
    if drug_info and drug_info["interactions"]:
        drug_context = "\n\nDRUG INTERACTION ALERT:\n"
        for interaction in drug_info["interactions"]:
            drug_context += f"- {interaction['description']} (Severity: {interaction['severity']})\n"

        retrieved.insert(
            0,
            {
                "text": drug_context,
                "metadata": {
                    "source_organization": "RxNorm/NIH",
                    "guideline_title": "Drug Interaction Database",
                    "page_number": "N/A",
                },
            },
        )

    if pubmed_results:
        pubmed_context = "\n\nRECENT RESEARCH (PubMed):\n"
        for article in pubmed_results:
            pubmed_context += (
                f"- {article['title']} ({article['journal']}, {article['pub_date']}). "
                f"URL: {article['url']}\n"
            )
        retrieved.append(
            {
                "text": pubmed_context,
                "metadata": {
                    "source_organization": "PubMed/NCBI",
                    "guideline_title": "Recent Research",
                    "page_number": "N/A",
                },
            }
        )

    response = generate_answer(
        query=query,
        retrieved_chunks=retrieved,
        prompt_version=prompt_version,
        model=model,
    )

    response["tools_used"] = tools_used
    response["drug_interactions"] = drug_info
    response["pubmed_articles"] = pubmed_results
    response["intent"] = intent
    return response
