from dataclasses import dataclass, replace
from typing import Literal

Retrieval = Literal["dense", "hybrid"]
VerifierAction = Literal["off", "annotate", "drop_unsupported"]


@dataclass(frozen=True)
class RunConfig:
    id: str
    label: str

    # Ladder axis
    retrieval: Retrieval = "dense"
    rerank: bool = False
    require_citations: bool = False
    verifier: VerifierAction = "off"
    abstain: bool = False

    # held constant across the whole ladder
    chunk_tokens: int = 400
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_k_context: int = 6
    rrf_k: int = 60
    temperature: float = 0.2
    agreement_samples: int = 3
    # Only consulted when abstain=True. Uncalibrated: picking a threshold
    # that corresponds to a real error rate is a separate step that needs
    # the eval grid to have been run first.
    abstain_threshold: float = 0.5
    generator_model: str = "gemini-3.6-flash"
    verifier_model: str = "openai/gpt-oss-120b"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


A = RunConfig(id="A", label="dense retrieval only, no abstention")
B = replace(A, id="B", label="+ hybrid retrieval (BM25 + dense, RRF)", retrieval="hybrid")
C = replace(B, id="C", label="+ cross-encoder reranker", rerank=True)
D = replace(C, id="D", label="+ mandatory inline citations", require_citations=True)
E = replace(D, id="E", label="+ groundedness verifier", verifier="drop_unsupported")
F = replace(E, id="F", label="+ calibrated abstention (full system)", abstain=True)

LADDER = (A, B, C, D, E, F)
