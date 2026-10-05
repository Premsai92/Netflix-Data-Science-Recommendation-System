
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("netflix_titles.csv")

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

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[-11:][::-1]

    similar_indices = similar_indices[similar_indices != idx][:10]

    return df["title"].iloc[similar_indices].tolist()


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
