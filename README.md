# Book Recommendation System (Hybrid)

This is a fully self-contained Book Recommendation System implemented in a single Jupyter notebook using a **custom-built dataset** of 20 books and 30+ user ratings. The system combines both **Content-Based Filtering** and **Collaborative Filtering** into a simple **Hybrid Recommender**.

---

## Features

- Content-Based Filtering (TF-IDF on book descriptions)
- Collaborative Filtering (user-book rating matrix)
- Hybrid Model (weighted average of content + collaborative scores)
- Fully offline, custom dataset — no external files needed
- Notebook-friendly and beginner-friendly design

---

## Dataset

The notebook generates a small dataset:

- `books_df`: 20 books with fields: `book_id`, `title`, `author`, `genre`, `description`
- `ratings_df`: 30+ sample ratings from multiple users (1-5 scale)

You can easily extend this with more books or real data.

---

## Tech Stack

- Python 3.11+
- `pandas`
- `numpy`
- `scikit-learn`

Runs seamlessly on Google Colab or any Jupyter Notebook environment.

---

## How It Works

### 🔍 Content-Based Filtering
- Uses TF-IDF vectorizer on book descriptions.
- Computes cosine similarity between all books.
- Returns top N books most similar in content to the input title.

### Collaborative Filtering
- Creates a user-book rating matrix.
- Computes cosine similarity between books based on user rating patterns.
- Returns top N books that users with similar preferences also liked.

### Hybrid Model
- Combines both similarity scores with a weighted parameter `alpha`:

```python
hybrid_score = alpha * content_similarity + (1 - alpha) * collaborative_similarity
```

- `alpha = 0.5` balances both equally
- Returns top N books ranked by combined score

---

## Usage

### Run in Notebook:
```python
recommend_content("Atomic Habits", top_n=5)
recommend_collab(3, top_n=5)  # book_id = 3
recommend_hybrid("Atomic Habits", top_n=5, alpha=0.5)
```

### Sample Output:
```
 Content-based Recommendations for 'Atomic Habits':
- The Power of Habit
- Deep Work
- The Subtle Art of Not Giving a F*ck
...
```

---

## Extensions

- Add more books and user ratings
- Replace TF-IDF with transformer embeddings (e.g. Sentence-BERT)
- Use matrix factorization (SVD, NMF) for collaborative filtering
- Evaluate using precision@k, recall@k, NDCG
- Create a Streamlit UI to make it interactive

 
Created by - Avneet Singh Oberoi (229310248)
Feel free to fork, extend, and experiment!

