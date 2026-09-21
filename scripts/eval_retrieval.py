from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from refuses_to_lie.pipeline import load_corpus_chunks
from refuses_to_lie.retrieval import Index

ROOT = Path(__file__).resolve().parent.parent
K_VALUES = (5, 10)


@dataclass
class QueryCase:
    query: str
    doc_id: str
    section_no: str


CASES = [
    QueryCase(
        "What happens to my NHS pension if I take a career break?",
        "a36-employment-break-policy-april-26",
        "10.1",
    ),
    QueryCase(
        "Do I get paid annual leave while I'm on a career break?",
        "a36-employment-break-policy-april-26",
        "6",
    ),
    QueryCase(
        "Can I appeal if my career break application is refused?",
        "a36-employment-break-policy-april-26",
        "12",
    ),
    QueryCase(
        "Can unused annual leave be carried into the next year?",
        "UHN-PO-HR24",
        "4.8",
    ),
    QueryCase(
        "What if I get sick while I'm on annual leave?",
        "UHN-PO-HR24",
        "4.9",
    ),
    QueryCase(
        "Who is responsible for making sure this policy is applied in practice?",
        "UHN-PO-HR24",
        "5",
    ),
    QueryCase(
        "Who qualifies for Statutory Sick Pay?",
        "Print Statutory Sick Pay (SSP) - GOV.UK",
        "3",
    ),
    QueryCase(
        "What am I entitled to when I take time off to have a baby?",
        "Print Maternity pay and leave - GOV.UK",
        "1",
    ),
    QueryCase(
        "What leave can I get if my partner is having a baby?",
        "Print Paternity pay and leave - GOV.UK",
        "1",
    ),
    QueryCase(
        "How much is Paternity Pay per week?",
        "Print Paternity pay and leave - GOV.UK",
        "3",
    ),
    QueryCase(
        "Can staff choose to retire gradually instead of all at once?",
        "uhn-flexible-retirement-policy-mos-20112023pdf",
        "1",
    ),
]


def rank_of_target(hits, doc_id: str, section_no: str) -> int | None:
    for i, hit in enumerate(hits, start=1):
        if hit.chunk.doc_id == doc_id and hit.chunk.section_no == section_no:
            return i
    return None


def main() -> None:
    print("Loading corpus...")
    chunks = load_corpus_chunks(ROOT / "corpus" / "employer", ROOT / "corpus" / "statutory")
    print(f"{len(chunks)} chunks across the corpus")

    print("Building index (BM25 + dense embeddings)...")
    index = Index(chunks)

    max_k = max(K_VALUES)
    results: dict[str, list[tuple[QueryCase, int | None]]] = {"dense": [], "hybrid": []}

    print()
    header = f"{'query':<62} {'dense rank':>10} {'hybrid rank':>12}"
    print(header)
    print("-" * len(header))
    for case in CASES:
        dense_hits = index.search_dense(case.query, k=max_k)
        hybrid_hits = index.search_hybrid(case.query, k=max_k)
        dense_rank = rank_of_target(dense_hits, case.doc_id, case.section_no)
        hybrid_rank = rank_of_target(hybrid_hits, case.doc_id, case.section_no)
        results["dense"].append((case, dense_rank))
        results["hybrid"].append((case, hybrid_rank))
        print(
            f"{case.query[:60]:<62} {dense_rank if dense_rank else '-':>10} "
            f"{hybrid_rank if hybrid_rank else '-':>12}"
        )

    print()
    n = len(CASES)
    for method in ("dense", "hybrid"):
        print(f"{method}:")
        for k in K_VALUES:
            hits_at_k = sum(1 for _, rank in results[method] if rank is not None and rank <= k)
            print(f"  recall@{k}: {hits_at_k}/{n} ({hits_at_k / n:.0%})")


if __name__ == "__main__":
    main()
