"""MinHash signature computation for related-tweet search."""

import numpy as np
from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
SHINGLE_K = 3


def _shingle(text: str) -> set[str]:
    """Produce word-level k-shingles from normalized text."""
    words = text.lower().split()
    if len(words) < SHINGLE_K:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + SHINGLE_K]) for i in range(len(words) - SHINGLE_K + 1)}


def compute_signature(ocr_text: str) -> bytes | None:
    """Compute a MinHash signature from OCR text. Returns serialized bytes, or None if text is empty."""
    if not ocr_text or not ocr_text.strip():
        return None
    text = " ".join(ocr_text.split())  # collapse all whitespace
    shingles = _shingle(text)
    if not shingles:
        return None
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m.hashvalues.tobytes()


def signature_to_minhash(sig_bytes: bytes) -> MinHash:
    """Deserialize a stored signature back into a MinHash object."""
    m = MinHash(num_perm=NUM_PERM)
    m.hashvalues = np.frombuffer(sig_bytes, dtype=np.uint64).copy()
    return m


LSH_THRESHOLD = 0.2


def build_lsh_index(rows):
    """Build LSH index from screenshot rows with minhash signatures.

    Returns (lsh_index, {id: MinHash} dict).
    """
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}
    for row in rows:
        m = signature_to_minhash(row["minhash_signature"])
        minhashes[row["id"]] = m
        lsh.insert(str(row["id"]), m)
    return lsh, minhashes


def query_related(lsh, minhashes, query_id, top_n=20):
    """Find related items. Returns list of (id, similarity) sorted by similarity desc."""
    if query_id not in minhashes:
        return []
    query_m = minhashes[query_id]
    candidates = lsh.query(query_m)
    results = []
    for cand_str in candidates:
        cand_id = int(cand_str)
        if cand_id == query_id:
            continue
        sim = query_m.jaccard(minhashes[cand_id])
        results.append((cand_id, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
