import h5py
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sentence_transformers import SentenceTransformer

from transformers import pipeline

# Data + Embeddings

MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"


def hash_st(st):
    return MODEL_NAME


@st.cache(hash_funcs={SentenceTransformer: hash_st})
def get_embedder():
    return SentenceTransformer(MODEL_NAME)


@st.cache
def embed_dataset(texts):
    embedder = get_embedder()
    embeddings = embedder.encode(texts)
    return embeddings


@st.cache
def sem_index_dir():
    path = "data/bert_index"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@st.cache
def embed_or_load_dataset(texts, version):
    index_path = os.path.join(sem_index_dir(), f"{version}.h5")

    if not os.path.exists(index_path):
        embeddings = embed_dataset(texts)
        with h5py.File(index_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings)

    with h5py.File(index_path, "r") as h:
        emb = np.array(h["embeddings"])
        return emb


@st.cache
def read_data(bible_csv, book_csv):
    book_df = pd.read_csv(book_csv)
    verses_df = pd.read_csv(bible_csv)
    df = pd.merge(verses_df, book_df, on="b")

    df = df[["id", "n", "c", "v", "t_x"]]

    col_rename = {"n": "book", "c": "chapter", "v": "verse", "t_x": "text"}
    df = df.rename(columns=col_rename)

    df["source"] = df.apply(
        lambda row: f"{row['book']} {row['chapter']}:{row['verse']}", axis=1
    )

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
    df = df.sort_values("score", ascending=False).reset_index()
    df = df[["source", "text", "score"]]
    return df


# Config
def config():
    # Static configs
    ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
    version_map = {
        "KJV": "t_kjv.csv",
        "ASV": "t_asv.csv",
    }

    # Configurable elements
    st.sidebar.subheader("Search config")
    version = st.sidebar.selectbox("Bible Version:", list(version_map.keys()))
    n_results = st.sidebar.number_input(
        "# Results", min_value=1, max_value=None, value=30
    )

    st.sidebar.subheader("Debug config")
    max_texts_limit = st.sidebar.number_input(
        "Limit text data to N (set to 0 if you want to use all texts):",
        min_value=0,
        max_value=None,
        value=0,
    )

    return ROOT_DIR, version, version_map, max_texts_limit, n_results


# Main


def main():
    ROOT_DIR, version, version_map, max_texts_limit, n_results = config()

    # Load Data
    bible_path = ROOT_DIR / f"data/{version_map[version]}"
    book_path = ROOT_DIR / "data/key_english.csv"
    search_df = read_data(bible_path, book_path)

    # Embed texts
    texts = search_df["text"].tolist()
    if max_texts_limit:
        texts = texts[:max_texts_limit]
    embeddings = embed_or_load_dataset(texts, version=version)

    # Search App
    st.title("What does the Bible say about ... ?")
    query = st.text_input("", "")

    if query:
        with st.spinner(f"Searching ..."):
            results = search(query, texts, embeddings, return_df=True, n=n_results)
            results = format_results(results, search_df)
            st.table(results)


if __name__ == "__main__":
    main()
