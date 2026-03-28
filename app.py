import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="📚 Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CACHING FOR PERFORMANCE
# ============================================================================


@st.cache_data
def load_and_process_data():
    """Load and preprocess data - matches notebook preprocessing"""
    df = pd.read_csv("data/books.csv", on_bad_lines="skip", engine="python")
    df.columns = df.columns.str.strip()

    # Handle missing values
    df["authors"] = df["authors"].fillna("Unknown")
    df["language_code"] = df["language_code"].fillna("unknown")
    df["title"] = df["title"].fillna("Unknown")
    df["average_rating"] = df["average_rating"].fillna(df["average_rating"].median())
    df["ratings_count"] = df["ratings_count"].fillna(0)
    df["text_reviews_count"] = df["text_reviews_count"].fillna(0)

    if "num_pages" in df.columns:
        df["num_pages"] = df["num_pages"].fillna(df["num_pages"].median())

    # Remove duplicates
    df = df.drop_duplicates(subset=["title", "authors"])

    # CRITICAL: Reset index so it matches similarity_matrix row positions
    df = df.reset_index(drop=True)

    # Convert data types
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
    df["text_reviews_count"] = pd.to_numeric(df["text_reviews_count"], errors="coerce")

    # Create combined text features for TF-IDF
    df["text_features"] = (
        df["title"].str.lower()
        + " "
        + df["authors"].str.lower()
        + " "
        + df["language_code"].str.lower()
    )

    # Add quality score (normalized rating for display)
    df["quality_score"] = (df["average_rating"] - df["average_rating"].min()) / (
        df["average_rating"].max() - df["average_rating"].min()
    )

    df["idx"] = range(len(df))

    return df


@st.cache_data
def build_recommendation_model(df):
    """Build TF-IDF model with custom stop words - matches notebook exactly"""
    model_path = "models/recommendation_model.pkl"

    # Try to load existing model
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            st.warning(f"Could not load cached model: {e}. Rebuilding...")

    # Build model if not found
    # Get English stop words and add custom patterns (same as notebook)
    stop_words = set(text.ENGLISH_STOP_WORDS)
    # Keep numbers and series info (legitimate book identifiers)
    # Remove only metadata words
    custom_stops = {
        "ed",
        "edition",
        "isbn",
        "vol",
        "volume",
    }
    stop_words = list(stop_words.union(custom_stops))

    # Initialize and fit TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000, stop_words=stop_words, ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["text_features"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Prepare model data
    model_data = {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "similarity_matrix": similarity_matrix,
        "df_clean": df,
    }

    # Save model for future use
    try:
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
    except Exception as e:
        st.warning(f"Could not save model: {e}")

    return model_data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def find_matching_books(query, df):
    """Find books matching user query (title or author) - returns matches with original indices"""
    query_lower = query.lower().strip()

    # Search in titles
    title_matches = df[df["title"].str.lower().str.contains(query_lower, na=False)]

    # Search in authors
    author_matches = df[df["authors"].str.lower().str.contains(query_lower, na=False)]

    # Combine and remove duplicates (keep original index)
    matches = (
        pd.concat([title_matches, author_matches])
        .drop_duplicates(subset=["title"])
        .sort_values("average_rating", ascending=False)
    )

    return matches


def get_recommendations(book_idx, n_recommendations=5, model_data=None):
    """Get recommendations for a book using its index"""
    try:
        df = model_data["df_clean"]
        similarity_matrix = model_data["similarity_matrix"]

        if book_idx >= len(similarity_matrix) or book_idx < 0:
            return None

        similarity_scores = similarity_matrix[book_idx]
        similar_indices = np.argsort(similarity_scores)[::-1][1 : n_recommendations + 1]

        rec_cols = [
            "title",
            "authors",
            "average_rating",
            "quality_score",
            "language_code",
        ]

        recommendations = df.iloc[similar_indices][rec_cols].copy()
        recommendations["similarity_score"] = similarity_scores[similar_indices]
        recommendations = recommendations.reset_index(drop=True)
        recommendations["rank"] = range(1, len(recommendations) + 1)

        return recommendations

    except Exception as e:
        print(f"ERROR in get_recommendations: {e}")
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# MAIN APP
# ============================================================================


@st.cache_resource
def init_model():
    """Initialize model - load from pickle or build if missing"""
    model_path = "models/recommendation_model.pkl"

    # Try to load existing model first (fastest path)
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            st.warning(f"Could not load model: {e}. Rebuilding...")

    # If no model, build from scratch
    df = load_and_process_data()
    model_data = build_recommendation_model(df)
    return model_data


def main():
    # Header
    st.markdown("# 📚 Intelligent Book Recommender System")

    # Load model (loads from cache if exists, otherwise builds)
    model_data = init_model()

    # Use the dataframe from model_data
    df = model_data["df_clean"]

    # Sidebar info
    with st.sidebar:
        st.markdown("## 📊 Dataset Info")
        st.metric("Total Books", len(df))
        st.metric("Avg Rating", f"{df['average_rating'].mean():.2f}")
        st.metric("Languages", df["language_code"].nunique())
        st.metric("Authors", df["authors"].nunique())

    # Main content
    st.markdown("---")

    # User input
    user_input = st.text_input(
        "📚 Enter a book title or author name:",
        placeholder="E.g., Dune, Harry Potter, Stephen King...",
        key="book_input",
    )

    if user_input.strip():
        # Find matching books (using model's dataframe)
        matches = find_matching_books(user_input, df)

        if len(matches) == 0:
            st.warning(f"❌ No books found matching '{user_input}'")
        else:
            # Show the matched book
            matched_book = matches.iloc[0]
            st.success(
                f"✅ Found: **{matched_book['title']}** by {matched_book['authors']}"
            )

            # Get the position of the matched book in the dataframe
            book_position = matched_book.name

            # Verify it's valid
            if book_position >= len(df):
                st.error(f"❌ Book index out of range")
            else:
                # Get recommendations using the book's position
                recs = get_recommendations(
                    book_position, n_recommendations=5, model_data=model_data
                )

                if recs is not None and len(recs) > 0:
                    st.markdown("---")
                    st.markdown("### 🎁 Top 5 Similar Books")

                    # Display each recommendation
                    for _, row in recs.iterrows():
                        col1, col2, col3 = st.columns([0.5, 3, 1.5])
                        with col1:
                            st.markdown(f"**#{int(row['rank'])}**")
                        with col2:
                            st.markdown(f"**{row['title'][:50]}**")
                            st.caption(f"by {row['authors'][:45]}")
                        with col3:
                            st.metric("Rating", f"{row['average_rating']:.1f}")

                        st.caption(f"📊 Match: {row['similarity_score']:.1%}")
                        st.divider()
                else:
                    st.error("❌ Could not generate recommendations. Please try again.")


if __name__ == "__main__":
    main()
