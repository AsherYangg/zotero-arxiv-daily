import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0', boost_keywords:str=None, boost_weight:float=5.0) -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    
    # Calculate days difference for each paper
    now = datetime.utcnow()
    dates = [datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ') for x in corpus]
    days_diff = np.array([(now - d).days for d in dates])
    days_diff = np.maximum(days_diff, 0) # Ensure no negative values

    # Exponential decay with 30-day half-life
    half_life = 30
    decay_rate = np.log(2) / half_life
    time_decay_weight = np.exp(-decay_rate * days_diff)
    time_decay_weight = time_decay_weight / time_decay_weight.sum()

    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]

    # --- Keyword Boosting ---
    if boost_keywords:
        # 1. Parse keywords
        keywords = [k.strip() for k in boost_keywords.split(',') if k.strip()]
        if keywords:
            # 2. Compute keyword embeddings
            keyword_features = encoder.encode(keywords)
            
            # 3. Compute similarity: [n_candidate, n_keywords]
            keyword_sim = encoder.similarity(candidate_feature, keyword_features)
            
            # 4. Take max similarity (match any keyword)
            # encoder.similarity returns a Tensor. .max(dim=1) returns (values, indices)
            max_keyword_scores = keyword_sim.max(dim=1).values
            
            # 5. Add weighted score
            scores += max_keyword_scores * boost_weight
    # ------------------------

    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate