from fastapi import FastAPI
from pathlib import Path

from src.models.candidate_gen import CandidateGenerator
from src.models.ranker import Ranker

app = FastAPI()

project_root = Path(__file__).resolve().parents[2]
PRE_DIR = project_root / "data" / "preprocessed"

cg = CandidateGenerator(pre_dir=PRE_DIR)
ranker = Ranker(pre_dir=PRE_DIR)
ranker.load()

@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 5):
    df = ranker.score_candidates(user_id, cg)
    top_k = df.head(k)

    return {
        "user_id": user_id,
        "recommendations": [
            {"paper_id": int(row.paper_id), "score": float(row.score)}
            for _, row in top_k.iterrows()
        ]
    }

