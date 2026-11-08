"""
Phase 7: Prior Case Retrieval (PCR)
Multi-Channel Breakthrough Retrieval Pipeline
Combines:
1. Enforced Self-Match Masking (S(q, c) = -inf if q == c)
2. Full-Text Sublinear TF-IDF (1, 2)
3. Indian Kanoon Citator & Cross-Citation Graph Matching
4. Statutory Provision Jaccard Alignment
5. Temporal Admissibility Filtering
6. LoRMA Dense Bi-Encoder Representation
"""
import math
import re
from collections import Counter
from typing import Dict, List, Set, Any
import numpy as np

def extract_statutory_provisions(text: str) -> Set[str]:
    """Extracts statutory sections, articles, and acts."""
    text_lower = text.lower()
    provisions = set()
    sec_matches = re.findall(r'(?:section|sec\.?|s\.)\s*(\d+[a-z]?(?:\(\d+[a-z]?\))*)', text_lower)
    for m in sec_matches:
        provisions.add(f"sec_{m}")
    art_matches = re.findall(r'(?:article|art\.?)\s*(\d+[a-z]?)', text_lower)
    for m in art_matches:
        provisions.add(f"art_{m}")
    return provisions

def compute_statutory_jaccard(prov_q: Set[str], prov_c: Set[str]) -> float:
    if not prov_q or not prov_c:
        return 0.0
    intersection = len(prov_q.intersection(prov_c))
    union = len(prov_q.union(prov_c))
    return float(intersection) / float(union) if union > 0 else 0.0

def build_citator_graph(docs: Dict[str, str]) -> Dict[str, Set[str]]:
    """Builds citation adjacency from Indian Kanoon CITATOR INFO headers and citations."""
    citations = {}
    for doc_id, text in docs.items():
        doc_cits = set()
        air_matches = re.findall(r'(?:19\d{2}|20\d{2})\s+air\s+(?:sc\s+)?\d+', text.lower())
        scc_matches = re.findall(r'\((?:19\d{2}|20\d{2})\)\s+\d+\s+scc\s+\d+', text.lower())
        scr_matches = re.findall(r'\[(?:19\d{2}|20\d{2})\]\s+\d+\s+scr\s+\d+', text.lower())
        doc_cits.update(air_matches)
        doc_cits.update(scc_matches)
        doc_cits.update(scr_matches)
        citations[doc_id] = doc_cits
    return citations

def mask_self_matches(sim_matrix: np.ndarray, query_ids: List[str], candidate_ids: List[str]) -> np.ndarray:
    """Enforces S(q, c) = -inf whenever candidate == query."""
    cand_id_map = {cid: idx for idx, cid in enumerate(candidate_ids)}
    masked_sims = sim_matrix.copy()
    for q_idx, qid in enumerate(query_ids):
        if qid in cand_id_map:
            c_idx = cand_id_map[qid]
            masked_sims[q_idx, c_idx] = -1e9
    return masked_sims
