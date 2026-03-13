"""
HotpotQA nDCG scoring for retrieval evaluation.

Supports two relevance modes:
- title: binary relevance from document title matching supporting_facts["title"]
- sentence (graded): 2 = doc contains a supporting sentence, 1 = same title only, 0 = otherwise
"""
import math
import re


def normalize_title(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def normalize_text(text):
    """Normalize text for sentence matching (collapse whitespace, strip, lower)."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def get_gold_titles(item):
    md = item.metadata or {}
    sf = md.get("supporting_facts", {})
    titles = sf.get("title", [])
    return {normalize_title(t) for t in titles}


def get_gold_supporting_sentences(item):
    """
    Extract gold supporting sentence strings from HotpotQA metadata.
    Uses metadata["context"] and metadata["supporting_facts"] to resolve
    (title, sent_id) -> sentence text.
    """
    md = item.metadata or {}
    ctx = md.get("context", {})
    sf = md.get("supporting_facts", {})
    sf_titles = sf.get("title", [])
    sf_sent_ids = sf.get("sent_id", [])

    ctx_titles = ctx.get("title", [])
    ctx_sentences = ctx.get("sentences", [])

    # Build lookup: index i -> (normalized_title, sentences list)
    title_to_sents = {}
    for i, t in enumerate(ctx_titles):
        if i < len(ctx_sentences):
            title_to_sents[normalize_title(t)] = ctx_sentences[i]

    gold_sentences = set()
    for t, sid in zip(sf_titles, sf_sent_ids):
        t_norm = normalize_title(t)
        if t_norm in title_to_sents and sid < len(title_to_sents[t_norm]):
            sent = title_to_sents[t_norm][sid]
            gold_sentences.add(normalize_text(sent))
    return gold_sentences


def extract_doc_title(doc):
    if "title" in doc and doc["title"]:
        return doc["title"]
    if "contents" in doc and doc["contents"]:
        return doc["contents"].split("\n")[0].strip().strip('"')
    return ""


def extract_doc_contents(doc):
    """Get full document text for sentence-containment checks."""
    if "contents" in doc and doc["contents"]:
        return doc["contents"]
    return ""


def dcg_at_k(rels, k):
    rels = rels[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(rels, k):
    ideal_rels = sorted(rels, reverse=True)
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def score_item_ndcg(
    item,
    docs,
    k=5,
    dedup=False,
    relevance_mode="sentence",
):
    """
    Compute nDCG@k for HotpotQA retrieval.

    Parameters
    ----------
    item : dict
        Dataset item with metadata["supporting_facts"] and metadata["context"].
    docs : list of dict
        Retrieved documents (each with "title" and/or "contents").
    k : int
        Top-k for nDCG.
    dedup : bool
        If True, remove repeated titles before truncating to k.
    relevance_mode : str
        - "title": Binary relevance from title match (legacy).
        - "sentence": Binary relevance from doc containing a supporting sentence.
        - "graded": Graded relevance: 2 = contains supporting sentence,
          1 = same title only, 0 = otherwise.

    Methodology (sentence / graded)
    --------------------------------
    Gold supporting sentences are extracted from metadata["context"] using
    metadata["supporting_facts"]["title"] and ["sent_id"]. A retrieved document
    is considered evidence-relevant (gain=2 in graded mode) if its contents
    contain any of those sentences (normalized substring match). In graded mode,
    documents that only match a supporting title but not a sentence get gain=1.
    """
    gold_titles = get_gold_titles(item)
    gold_sentences = get_gold_supporting_sentences(item) if relevance_mode in ("sentence", "graded") else set()

    retrieved_titles = []
    retrieved_contents = []
    seen = set()

    for doc in docs:
        title = normalize_title(extract_doc_title(doc))
        if dedup:
            if title in seen:
                continue
            seen.add(title)
        retrieved_titles.append(title)
        retrieved_contents.append(extract_doc_contents(doc))

    retrieved_titles = retrieved_titles[:k]
    retrieved_contents = retrieved_contents[:k]

    if relevance_mode == "title":
        rels = [1 if title in gold_titles else 0 for title in retrieved_titles]
    elif relevance_mode == "sentence":
        rels = []
        for contents in retrieved_contents:
            cnt_norm = normalize_text(contents)
            has_supporting = any(gs in cnt_norm for gs in gold_sentences)
            rels.append(1 if has_supporting else 0)  # binary at evidence level
    else:  # graded
        rels = []
        for title, contents in zip(retrieved_titles, retrieved_contents):
            cnt_norm = normalize_text(contents)
            has_supporting = any(gs in cnt_norm for gs in gold_sentences)
            title_match = title in gold_titles
            if has_supporting:
                rels.append(2)
            elif title_match:
                rels.append(1)
            else:
                rels.append(0)

    score = ndcg_at_k(rels, k)

    return {
        "ndcg": score,
        "rels": rels,
        "gold_titles": gold_titles,
        "gold_sentences": gold_sentences if relevance_mode in ("sentence", "graded") else None,
        "retrieved_titles": retrieved_titles,
    }
