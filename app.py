import json
import os
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st
from sentence_transformers import SentenceTransformer

# Data + Embeddings


def get_embedder():
    return SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")


@st.cache
def embed_dataset(texts):
    embedder = get_embedder()
    embeddings = embedder.encode(texts)
    return embeddings


@st.cache
def read_data(bible_csv, book_csv):
    book_df = pd.read_csv(book_csv)
    verses_df = pd.read_csv(bible_csv)
    df = pd.merge(verses_df, book_df, on="b")

    df = df[["id", "n", "c", "v", "t_x"]]

    col_rename = {"n": "book", "c": "chapter", "v": "verse", "t_x": "text"}
    df = df.rename(columns=col_rename)

    return df


# Search


def search(query, texts, embeddings, n=None, return_df=False):
    query = get_embedder().encode([query])[0]
    sims = sklearn.metrics.pairwise.cosine_similarity([query], embeddings)[0]
    results = sorted(list(zip(texts, sims)), key=lambda x: x[1], reverse=True)
    if n:
        results = results[:n]

    if return_df:
        results = pd.DataFrame(results, columns=["text", "score"])
    return results


def format_results(results_df, search_df):
    df = pd.merge(results_df, search_df, on="text")
    df = df.sort_values("score", ascending=False)
    df = df[["book", "chapter", "verse", "text", "score"]]
    return df


# Main


def main():
    # Static configs
    ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
    version_map = {
        "KJV": "t_kjv.csv",
        "ASV": "t_asv.csv",
    }
    MAX_TEXTS_LIMIT = None  # Embedding may be too taxing on your machine. Set this if you want to limit the number of texts.
    N_RESULTS = 50

    # Version Picker + Load Data
    version = st.selectbox("Bible Version:", list(version_map.keys()))
    bible_path = ROOT_DIR / f"data/{version_map[version]}"
    book_path = ROOT_DIR / "data/key_english.csv"
    search_df = read_data(bible_path, book_path)

    # Embed texts
    texts = search_df["text"].tolist()
    if MAX_TEXTS_LIMIT:
        texts = texts[:MAX_TEXTS_LIMIT]
    st.subheader(f"Embedding {len(texts)} texts from {version}...")
    bar = st.progress(0)
    embeddings = embed_dataset(texts)
    bar.progress(100)

    # Search App
    st.title("What does the Bible say about ... ?")
    query = st.text_input("", "")

    if query:
        results = search(query, texts, embeddings, return_df=True, n=N_RESULTS)
        final_df = format_results(results, search_df)
        st.table(final_df)


if __name__ == "__main__":
    main()
