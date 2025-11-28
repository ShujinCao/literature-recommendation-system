import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class CandidateGenerator:
    """
    Two-stage candidate generator for literature recommendation system.
    Loads preprocessed paper/user data and embeddings.
    """

    def __init__(
        self,
        pre_dir=None,
        papers_file="papers_preprocessed.csv",
        users_file="users_preprocessed.csv",
        emb_file="paper_embeddings.npy",
        mapping_file="paper_id_to_index.csv",
    ):

        # Auto-detect project root based on this file's location
        # candidate_gen.py → models → src → project_root
        project_root = Path(__file__).resolve().parents[2]
        print(project_root)

        # If user passes pre_dir, use it. Otherwise use auto-detected path.
        if pre_dir is None:
            self.pre_dir = project_root / "data" / "preprocessed"
        else:
            self.pre_dir = Path(pre_dir)

        print("[DEBUG] Looking for preprocessed data in:", self.pre_dir.resolve())
        # Load preprocessed data
        self.papers = pd.read_csv(self.pre_dir / papers_file)
        self.users = pd.read_csv(self.pre_dir / users_file)
        self.embeddings = np.load(self.pre_dir / emb_file)
        self.mapping = pd.read_csv(self.pre_dir / mapping_file)

        # Precompute topic → list of embedding indices
        self.topic_to_paper_idx = self._build_topic_index()

    def _build_topic_index(self):
        topic_dict = {}
        for idx, row in self.papers.iterrows():
            t = row["topic_primary"]
            topic_dict.setdefault(t, []).append(idx)
        return topic_dict

    def get_user_embedding(self, user_id):
        user_row = self.users[self.users["user_id"] == user_id]
        if user_row.empty:
            raise ValueError(f"User ID {user_id} not found.")

        user_row = user_row.iloc[0]
        t1, t2 = user_row["research_focus_1"], user_row["research_focus_2"]

        idxs = (
            self.topic_to_paper_idx.get(t1, []) +
            self.topic_to_paper_idx.get(t2, [])
        )

        if len(idxs) == 0:
            return self.embeddings.mean(axis=0)

        return self.embeddings[idxs].mean(axis=0)

    def get_top_n(self, user_id, n=200):
        user_emb = self.get_user_embedding(user_id).reshape(1, -1)
        sims = cosine_similarity(user_emb, self.embeddings)[0]

        top_idxs = np.argsort(sims)[::-1][:n]
        paper_ids = self.papers.iloc[top_idxs]["paper_id"].tolist()

        return paper_ids, sims[top_idxs].tolist()

