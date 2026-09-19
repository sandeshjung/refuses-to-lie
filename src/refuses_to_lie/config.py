from dataclasses import dataclass
from typing import Literal
from dataclasses import replace

Retrieval = Literal["dense", "hybrid"]
VerifierAction = Literal["off", "annotate", "drop_unsupported"]

@dataclass(frozen=True)     # frozen=True makes instances immutable
class RunConfig:
    id: str
    label: str

    # Ladder axis
    retrieval: str = "dense"
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
    generator_model: str = ""
    verifier_model: str = ""

A = RunConfig(id="A", label="dense retrieval only, no abstention")
B = replace(A, id="B", label="+ hybrid retrieval (BM25 + dense, RRF)", retrieval="hybrid")
C = replace(B, id="C", label="+ cross-encoder reranker", rerank=True)
D = replace(C, id="D", label="+ mandatory inline citations", require_citations=True)
E = replace(D, id="E", label="+ groundedness verifier", verifier="drop_unsupported")
F = replace(E, id="F", label="+ calibrated abstention (full system)", abstain=True)

LADDER = (A, B, C, D, E, F)