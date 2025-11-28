import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb


class Ranker:
    """
    Gradient-boosted ranking model for the second-stage reranker.
    Trains a LightGBM model using features derived from:
      - cosine similarity
      - paper recency
      - topic match
    """

    def __init__(
        self,
        pre_dir=None,
        model_dir=None,
        model_file="ranker.txt",
        papers_file="papers_preprocessed.csv",
        users_file="users_preprocessed.csv",
        interactions_file="interactions_preprocessed.csv",
    ):

        # Detect project structure
        project_root = Path(__file__).resolve().parents[2]

        # Preprocessed data directory
        self.pre_dir = Path(pre_dir) if pre_dir else (project_root / "data" / "preprocessed")

        # Model directory
        self.model_dir = Path(model_dir) if model_dir else (project_root / "models")
        self.model_dir.mkdir(exist_ok=True)

        self.model_file = self.model_dir / model_file

        # Load data
        self.papers = pd.read_csv(self.pre_dir / papers_file)
        self.users = pd.read_csv(self.pre_dir / users_file)
        self.interactions = pd.read_csv(self.pre_dir / interactions_file)

        # Feature columns
        self.feature_cols = ["similarity", "paper_recency", "topic_match"]

        # LightGBM model placeholder
        self.model = None


    # ---------------------------------------------------------
    # Helper: build training dataframe
    # ---------------------------------------------------------
    def build_training_data(self, candidate_generator, n_candidates=200):
        """
        Construct ranking features and labels for each user.
        """

        rows = []

        for user_id in self.users["user_id"].tolist():
            # Get candidates from stage 1
            paper_ids, sims = candidate_generator.get_top_n(user_id, n=n_candidates)

            df = pd.DataFrame({
                "user_id": user_id,
                "paper_id": paper_ids,
                "similarity": sims,
            })

            # Merge metadata
            df = df.merge(
                self.papers[["paper_id", "paper_recency", "topic_primary"]],
                on="paper_id",
                how="left"
            )

            # Topic match feature
            u = self.users[self.users.user_id == user_id].iloc[0]
            df["topic_match"] = (
                (df["topic_primary"] == u["research_focus_1"]) |
                (df["topic_primary"] == u["research_focus_2"])
            ).astype(int)

            # Label (1 = interacted)
            interacted = self.interactions[self.interactions.user_id == user_id]["paper_id"].tolist()
            df["label"] = df["paper_id"].isin(interacted).astype(int)

            rows.append(df)

        train_df = pd.concat(rows, ignore_index=True)
        return train_df


    # ---------------------------------------------------------
    # Train the ranking model
    # ---------------------------------------------------------
    def train(self, train_df):
        """
        Train a LightGBM LambdaRank model.
        """

        X = train_df[self.feature_cols]
        y = train_df["label"]

        # Group sizes = number of candidates per user
        group = train_df.groupby("user_id").size().tolist()

        train_data = lgb.Dataset(X, label=y, group=group)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "learning_rate": 0.05,
            "num_leaves": 32,
            "max_depth": -1,
        }

        print("Training LightGBM LambdaRank model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
        )

        # Save model
        self.model.save_model(str(self.model_file))
        print("Saved trained model to:", self.model_file)


    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    def load(self):
        """
        Load a saved LightGBM ranking model.
        """
        print("Loading model:", self.model_file)
        self.model = lgb.Booster(model_file=str(self.model_file))


    # ---------------------------------------------------------
    # Score candidates (for serving API)
    # ---------------------------------------------------------
    def score_candidates(self, user_id, candidate_generator):
        """
        Given a user and candidate generator, return sorted scores.
        """

        paper_ids, sims = candidate_generator.get_top_n(user_id, n=200)

        df = pd.DataFrame({
            "paper_id": paper_ids,
            "similarity": sims,
        })

        # Merge metadata
        df = df.merge(
            self.papers[["paper_id", "paper_recency", "topic_primary"]],
            on="paper_id",
            how="left"
        )

        # Topic match feature
        u = self.users[self.users.user_id == user_id].iloc[0]
        df["topic_match"] = (
            (df["topic_primary"] == u["research_focus_1"]) |
            (df["topic_primary"] == u["research_focus_2"])
        ).astype(int)

        X_user = df[self.feature_cols]

        # Ensure model is loaded
        if self.model is None:
            self.load()

        df["score"] = self.model.predict(X_user)

        # Sort papers by score
        df = df.sort_values("score", ascending=False)

        return df

