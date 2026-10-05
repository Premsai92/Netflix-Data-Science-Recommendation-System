
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("/content/netflix_titles.csv")

# Fill missing values
df["director"] = df["director"].fillna("")
df["cast"] = df["cast"].fillna("")
df["listed_in"] = df["listed_in"].fillna("")
df["description"] = df["description"].fillna("")

# Create content
df["content"] = (
    df["listed_in"] + " " +
    df["description"] + " " +
    df["cast"] + " " +
    df["director"]
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])

# Cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Create title index
indices = pd.Series(
    df.index,
    index=df["title"]
).drop_duplicates()

# Recommendation function
def recommend(title):
    idx = indices[title]

    similarity_scores = list(enumerate(similarity[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:11]

    movie_indices = [i[0] for i in similarity_scores]

    return df["title"].iloc[movie_indices].tolist()


# -------------------------
# WEBSITE
# -------------------------

st.title("🎬 Netflix Recommendation System")

st.subheader("Discover movies and TV shows you may like!")

st.write(
    "Select a movie or TV show below and our machine-learning "
    "model will recommend similar titles."
)

st.divider()

# Movie selection
selected_title = st.selectbox(
    "Choose a movie or TV show:",
    df["title"].sort_values().tolist()
)

# Selected movie information
selected_movie = df[df["title"] == selected_title].iloc[0]

st.subheader("🎬 Selected Title")

st.write("**Type:**", selected_movie["type"])
st.write("**Release Year:**", selected_movie["release_year"])
st.write("**Rating:**", selected_movie["rating"])
st.write("**Genre:**", selected_movie["listed_in"])
st.write("**Description:**", selected_movie["description"])

st.divider()

# Recommendation button
if st.button("🎯 Get Recommendations"):

    recommendations = recommend(selected_title)

    st.subheader("🍿 Recommended for You")

    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
