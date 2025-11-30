# literature-recommendation-system
A scalable research-literature recommendation system using document embeddings, semantic search, and ML ranking. Built to help researchers discover relevant studies based on their interests, past reading behavior, and topic profiles.

# Example API Call (FastAPI + curl)
### Start the server:
```
uvicorn src.serving.app:app --reload
```
### Then call the recommender:
```
curl "http://localhost:8000/recommend/10?k=5"
```
### Example response:
```
{
  "user_id": 10,
  "recommendations": [
    { "paper_id": 240, "score": 0.732971813523899 },
    { "paper_id": 257, "score": 0.6423691750872715 },
    { "paper_id": 856, "score": 0.6310004990360234 },
    { "paper_id": 588, "score": 0.6050107100810015 },
    { "paper_id": 129, "score": 0.5584257540714496 }
  ]
}
```
