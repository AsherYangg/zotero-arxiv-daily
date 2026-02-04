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
    
    # Convert to numpy array for consistent operations
    scores = np.array(scores, dtype=np.float32)

    # --- Keyword Boosting ---
    if boost_keywords:
        # 1. Parse keywords
        keywords = [k.strip() for k in boost_keywords.split(',') if k.strip()]
        if keywords:
            # 2. Ensure boost_weight is float (env vars are strings)
            boost_weight = float(boost_weight)
            
            # 3. Compute candidate embeddings using Title + Abstract for better keyword matching
            candidate_text = [f"{paper.title}. {paper.summary}" for paper in candidate]
            candidate_keyword_feature = encoder.encode(candidate_text)
            
            # 4. Compute keyword embeddings
            keyword_features = encoder.encode(keywords)
            
            # 5. Compute similarity: [n_candidate, n_keywords]
            keyword_sim = encoder.similarity(candidate_keyword_feature, keyword_features)
            
            # 6. Take max similarity (match any keyword)
            # encoder.similarity returns a Tensor. .max(dim=1) returns (values, indices)
            max_keyword_scores = keyword_sim.max(dim=1).values
            max_keyword_scores = np.array(max_keyword_scores, dtype=np.float32)
            
            # 7. Apply sigmoid scaling to prevent boost from overwhelming original ranking
            # sigmoid((sim - 0.5) * 10) maps:
            #   sim=0.3 -> ~0.12, sim=0.5 -> 0.5, sim=0.7 -> ~0.88, sim=0.8 -> ~0.95
            # This ensures smooth transition and caps the maximum boost at boost_weight
            scaled_boost = 1 / (1 + np.exp(-10 * (max_keyword_scores - 0.5)))
            
            # 8. Clamp to [0, 1] for safety, then apply weight
            scaled_boost = np.clip(scaled_boost, 0, 1)
            scores += scaled_boost * boost_weight
    # ------------------------

    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate