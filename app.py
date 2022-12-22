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

from src.retriever import Retriever

import urllib.parse

# Data + Embeddings

# MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"
MODEL_NAME = "msmarco-distilbert-base-v4"


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
def read_data(bible_csv, book_csv, agg_chapter=False):
    book_df = pd.read_csv(book_csv)
    verses_df = pd.read_csv(bible_csv)
    df = pd.merge(verses_df, book_df, on="b")

    df = df[["id", "n", "c", "v", "t_x"]]

    col_rename = {"n": "book", "c": "chapter", "v": "verse", "t_x": "text"}
    df = df.rename(columns=col_rename)

    df["source"] = df.apply(
        lambda row: f"{row['book']} {row['chapter']}:{row['verse']}", axis=1
    )

    if agg_chapter:
        df = df.groupby(["book", "chapter"])["text"].apply(" ".join).reset_index()
        df["source"] = df.apply(lambda row: f"{row['book']} {row['chapter']}", axis=1)
    else:
        df["source"] = df.apply(
            lambda row: f"{row['book']} {row['chapter']}:{row['verse']}", axis=1
        )

    return df


# Search
def search(query, texts, embeddings, n=None, return_df=False, threshold=0):
    query = get_embedder().encode([query])[0]
    sims = sklearn.metrics.pairwise.cosine_similarity([query], embeddings)[0]
    results = sorted(list(zip(texts, sims)), key=lambda x: x[1], reverse=True)
    if n:
        results = results[:n]

    if threshold:
        results = [x for x in results if x[1] >= threshold]

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
    per_verse_df = read_data(bible_path, book_path, agg_chapter=False)
    per_chapter_df = read_data(bible_path, book_path, agg_chapter=True)

    retriever = Retriever(per_chapter_df["text"].tolist())

    # Embed texts
    all_verses = per_verse_df["text"].tolist()
    if max_texts_limit:
        all_verses = all_verses[:max_texts_limit]
    embeddings = embed_or_load_dataset(all_verses, version=version)

    # Search App
    st.title("What does the Bible say about ... ?")
    query = st.text_input("", "")

    if query:
        with st.spinner(f"Searching ..."):

            # Retrieve based on character tf-idf
            verse_results = retriever.search(query, n_results=n_results)
            matches = per_chapter_df.iloc[verse_results.indices]
            matches["score"] = verse_results.data

            # RE-RANK ACCORDING TO SEMANTIC SEARCH

            all_results = []

            for idx, match in matches.iterrows():
                # Filter the embeddings and verses to the chapter match
                book_mask = per_verse_df["book"] == match["book"]
                chapter_mask = per_verse_df["chapter"] == match["chapter"]
                target_chapter_df = per_verse_df[book_mask & chapter_mask]

                # Get embeddings corresponding to the book/chapter verse indices
                target_embeddings = embeddings[target_chapter_df.index.tolist()]

                # Perform semantic search
                verse_results = search(
                    query,
                    target_chapter_df["text"].tolist(),
                    target_embeddings,
                    return_df=True,
                    n=5,
                    threshold=0.4,
                )
                max_score = verse_results["score"].max()

                # Highlight verses that meet the threshold
                formatted_text = match["text"]
                for _, text in verse_results.iterrows():
                    formatted_text = formatted_text.replace(
                        text["text"],
                        f'<span style="color:green">{text["text"]}</span>',
                    )

                # Do this only if
                # Create a dict to store the results
                biblegateway_url = urllib.parse.quote(
                    f"https://www.biblegateway.com/passage/?search={match['book']} {match['chapter']}&version=NIV"
                )
                result = {
                    "source": match["source"],
                    "url": biblegateway_url,
                    "score": max_score,
                    "formatted_text": formatted_text,
                    "raw_verse_results": verse_results,
                }

                if max_score > 0:
                    all_results.append(result)

            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

            if len(all_results) == 0:
                st.write("No results. Please try paraphrasing your query.")
            else:
                for idx, result in enumerate(all_results):
                    st.write("---")
                    # st.write(result)
                    # st.write(f"### Result {idx+1} / {len(all_results)}:")
                    st.write(
                        f"#### [Result {idx+1} / {len(all_results)}] [{result['source']}]({result['url']})"
                    )
                    st.markdown(result["formatted_text"], unsafe_allow_html=True)
                    st.table(result["raw_verse_results"])


if __name__ == "__main__":
    main()
