import ast
import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
@st.cache_data
def get_movie_poster(movie_title):
    if not OMDB_API_KEY:
        return None

    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={
                "apikey": OMDB_API_KEY,
                "t": movie_title
            },
            timeout=10
        )

        data = response.json()
        poster = data.get("Poster")

        if poster and poster != "N/A":
            return poster

    except requests.RequestException:
        return None

    return None
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b0b14, #19192b, #24152e);
    color: #ffffff;
}

h1, h2, h3, p, label {
    color: #ffffff !important;
}

.movie-card {
    background: #202035;
    padding: 16px;
    border-radius: 16px;
    min-height: 140px;
    border: 1px solid #393953;
}

div.stButton > button {
    background: linear-gradient(90deg, #e50914, #ff4b5c);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: #b80710;
    border: 1px solid #ffffff;
    transform: scale(1.02);
}

img {
    border-radius: 12px;
}

[data-testid="stCaptionContainer"] {
    color: #c4c4d8;
}
</style>
""", unsafe_allow_html=True)

st.html("""
<div style="
    text-align:center;
    padding:38px 15px;
    background:radial-gradient(ellipse at center, #31070d 0%, #12070b 55%, #08080d 100%);
    border:1px solid #ff3048;
    border-radius:22px;
    box-shadow:0 0 22px rgba(255,30,60,0.25);
    margin-bottom:28px;
">
    <p style="
        color:#f4c5cb;
        letter-spacing:5px;
        font-size:12px;
        font-weight:600;
        margin-bottom:18px;
    ">
        YOUR PERSONAL MOVIE UNIVERSE
    </p>

    <h1 style="
        font-size:clamp(42px, 7vw, 68px);
        font-weight:900;
        letter-spacing:1px;
        margin:0;
        line-height:1.2;
        color:#ffffff;
        text-shadow:0 0 18px rgba(255,30,60,0.25);
    ">
        🎬 Cine<span style="
            color:#ff3048;
            text-shadow:0 0 22px rgba(255,30,60,0.65);
        ">Match</span>
    </h1>

    <div style="
        width:85px;
        height:4px;
        background:#ff3048;
        border-radius:10px;
        margin:20px auto;
        box-shadow:0 0 12px #ff3048;
    "></div>

    <p style="
        font-size:20px;
        font-weight:600;
        color:#ffffff;
        margin-bottom:10px;
    ">
        Every movie has a match.
    </p>

    <p style="
        font-size:14px;
        color:#c5b9c0;
        margin-bottom:0;
    ">
        Discover films you'll love, powered by AI.
    </p>
</div>
""")

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    
    credits = pd.read_csv("tmdb_5000_credits.csv")

    credits = credits.rename(columns={"movie_id": "id"})
    movies = movies.merge(
        credits[["id", "cast", "crew"]],
        on="id",
        how="left"
    )

    def parse_list(value):
        try:
            items = ast.literal_eval(value)
            return [
                item["name"].lower().replace(" ", "")
                for item in items
            ]
        except (ValueError, TypeError, SyntaxError):
            return []

    movies["overview"] = movies["overview"].fillna("")
    movies["genres_list"] = movies["genres"].apply(parse_list)
    movies["keywords_list"] = movies["keywords"].apply(parse_list)

    movies = movies.drop_duplicates(subset="title")
    movies = movies.reset_index(drop=True)

    movies["tags"] = (
        movies["genres_list"].apply(lambda x: " ".join(x * 3))
        + " "
        + movies["keywords_list"].apply(lambda x: " ".join(x * 2))
        + " "
        + movies["overview"].str.lower()
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(movies["tags"])

    return movies, matrix

try:
    movies, matrix = load_data()

    st.success(f"Loaded {len(movies):,} movies into CineMatch.")

    selected_movie = st.selectbox(
        "🔎 Choose a movie",
        sorted(movies["title"].dropna().unique())
    )

    chosen = movies[movies["title"] == selected_movie].iloc[0]

    left, right = st.columns([1, 2])

    with left:
        poster = get_movie_poster(str(chosen["title"]))

        if poster:
            st.image(poster, width=230)
        else:
            st.info("Poster not available")

    with right:
        st.subheader(str(chosen["title"]))
        release_date = chosen.get("release_date", "")

        if pd.notna(release_date) and str(release_date).strip():
            st.caption(f"📅 Release Year: {str(release_date)[:4]}")
        rating = chosen.get("vote_average")
        if pd.notna(rating):
            st.write(f"⭐ TMDB Rating: {rating:.1f}/10")

        genres = chosen["genres_list"]
        st.write("🎭 Genres: " + ", ".join(g.title() for g in genres))
        st.write(str(chosen["overview"]) or "No description available.")
    if st.button("🍿 Discover My Next Movie", use_container_width=True):
        movie_index = chosen.name
        scores = cosine_similarity(
            matrix[movie_index], matrix
        ).flatten()

        candidates = movies.copy()
        candidates["similarity"] = scores
        candidates = candidates[candidates.index != movie_index]

        language = chosen.get("original_language")
        if language == "hi":
            candidates = candidates[
                candidates["original_language"] == "hi"
            ]

        recommendations = candidates.sort_values(
            "similarity", ascending=False
        ).head(5)

        st.divider()
        st.header("🍿 Recommended For You")

        if recommendations.empty:
            st.warning("No similar movies found.")
        else:
            cols = st.columns(5)
            for col, (_, movie) in zip(cols, recommendations.iterrows()):
                with col:
                    poster = get_movie_poster(str(movie["title"]))

                    if poster:
                        st.image(
                            poster,
                            use_container_width=True
                        )
                    else:
                        st.write("🎬 Poster unavailable")

                    st.markdown(f"**{movie['title']}**")

                    rating = movie.get("vote_average")
                    if pd.notna(rating):
                        st.caption(f"⭐ {rating:.1f}/10")

                    release_date = movie.get("release_date", "")
                    if pd.notna(release_date) and str(release_date).strip():
                        st.caption(f"📅 {str(release_date)[:4]}")

                    st.caption(
                        f"Similarity: {movie['similarity']:.3f}"
                    )

except FileNotFoundError:
    st.error(
        "Dataset files not found. Keep both CSV files in the same folder as app.py."
    )
except Exception as error:
    st.error(f"Something went wrong: {error}")
