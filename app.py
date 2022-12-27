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
def read_data(bible_csv, metadata_csv, agg_chapter=False):

    # There is one constant metadata file (metadata_csv),
    #   and another csv file containing the actual verses in the specified version (bible_csv).
    metadata_df = pd.read_csv(metadata_csv)
    verses_df = pd.read_csv(bible_csv, escapechar="\\")
    df = pd.merge(verses_df, metadata_df, on="b")
    df = df.fillna("")  # Some verses are blank in some versions

    df = df[["n", "c", "v", "t_x"]]

    # The data sources used have this convention in the columns.
    # Renaming them here for ease of remembrance.
    col_rename = {"n": "book", "c": "chapter", "v": "verse", "t_x": "text"}
    df = df.rename(columns=col_rename)

    # Create a human-friendly string of specifying a verse (e.g. Genesis 1:1)
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
    # df = df[["source", "chapter", "book" "text", "score"]]
    return df


def get_chapter(per_chapter_df, book, chapter):
    book_mask = per_chapter_df["book"] == book
    chapter_mask = per_chapter_df["chapter"] == chapter
    return per_chapter_df[book_mask & chapter_mask]


# Config
def config():
    # Static configs
    ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
    version_map = {
        "NIV": "niv.csv",
        "ESV": "esv.csv",
    }

    # Configurable elements
    st.sidebar.subheader("Search config")
    version = st.sidebar.selectbox("Bible Version:", list(version_map.keys()))
    n_results = st.sidebar.number_input(
        "# Results", min_value=1, max_value=None, value=30
    )

    st.sidebar.subheader("Debug config")

    debug_mode = st.sidebar.checkbox("Show debug logs?", value=True)

    max_texts_limit = st.sidebar.number_input(
        "Limit text data to N (set to 0 if you want to use all texts):",
        min_value=0,
        max_value=None,
        value=0,
    )

    # Tf-Idf Retriever
    use_tfidf_retriever = st.sidebar.checkbox("Tf-Idf Retriever", value=True)
    use_semantic_verse_retriever = st.sidebar.checkbox(
        "Semantic Verse Retriever", value=True
    )

    return (
        ROOT_DIR,
        version,
        version_map,
        max_texts_limit,
        n_results,
        debug_mode,
        use_tfidf_retriever,
        use_semantic_verse_retriever,
    )


# Main


def main():
    (
        ROOT_DIR,
        version,
        version_map,
        max_texts_limit,
        n_results,
        debug_mode,
        use_tfidf_retriever,
        use_semantic_verse_retriever,
    ) = config()

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
    query = st.text_input("", "is it ok for a believer to continue in sin?")

    if query:
        with st.spinner(f"Searching ..."):

            candidates = pd.DataFrame()
            tfidf_candidates = pd.DataFrame()
            semantic_candidates = pd.DataFrame()

            if use_tfidf_retriever:
                # Retrieve based on character tf-idf
                chapter_results = retriever.search(query, n_results=n_results)
                tfidf_candidates = per_chapter_df.iloc[chapter_results.indices]
                tfidf_candidates["score"] = chapter_results.data
                candidates = pd.concat([candidates, tfidf_candidates])
                if debug_mode:
                    st.header("Chapter Search (Tf-Idf per Chapter)")
                    st.write(tfidf_candidates[["source", "score"]])

            if use_semantic_verse_retriever:
                # Retrieve based on verse similarity
                # collect the chapters from the verse results and add them to the candidates

                verse_results = search(
                    query,
                    per_verse_df["text"].tolist(),
                    embeddings,
                    return_df=True,
                    n=n_results * 3,
                    threshold=0.5,
                )
                verse_results = format_results(verse_results, per_verse_df)
                if debug_mode:
                    st.header("Verse Search (Semantic Search per Verse)")
                    st.write(verse_results)

                # Generate chapters from verse results
                semantic_candidates = [
                    get_chapter(per_chapter_df, verse["book"], verse["chapter"])
                    for _, verse in verse_results.iterrows()
                ]
                semantic_candidates = pd.concat(semantic_candidates)
                candidates = pd.concat([candidates, semantic_candidates])

            # DEBUG: Print out chapters found by each retriever
            if debug_mode:
                tfidf_candidate_chapters = (
                    set(tfidf_candidates["source"].unique().tolist())
                    if use_tfidf_retriever
                    else set()
                )
                semantic_candidate_chapters = (
                    set(semantic_candidates["source"].unique().tolist())
                    if use_semantic_verse_retriever
                    else set()
                )

                common = tfidf_candidate_chapters.intersection(
                    semantic_candidate_chapters
                )
                tfidf_unique = tfidf_candidate_chapters - semantic_candidate_chapters
                semantic_unique = semantic_candidate_chapters - tfidf_candidate_chapters

                st.write(f"Common chapters found by both: {common}")
                st.write(f"Unique chapters found by Tf-Idf (Chapter): {tfidf_unique}")
                st.write(
                    f"Unique chapters found by Semantic Search (Verse): {semantic_unique}"
                )

            # Ensure no duplicate candiddate chapters
            candidates = candidates.drop_duplicates(subset=["book", "chapter"])
            if debug_mode:
                st.write(f"Candidate chapters: {sorted(candidates['source'].tolist())}")

            # RE-RANK ACCORDING TO SEMANTIC SEARCH

            all_results = []

            for idx, candidate in candidates.iterrows():
                # Filter the embeddings and verses to the chapter match
                book_mask = per_verse_df["book"] == candidate["book"]
                chapter_mask = per_verse_df["chapter"] == candidate["chapter"]
                target_chapter_df = per_verse_df[book_mask & chapter_mask]

                # Get embeddings corresponding to the book/chapter verse indices
                target_embeddings = embeddings[target_chapter_df.index.tolist()]

                # Perform semantic search per chapter
                chapter_results = search(
                    query,
                    target_chapter_df["text"].tolist(),
                    target_embeddings,
                    return_df=True,
                    n=5,
                    threshold=0.5,
                )
                max_score = chapter_results["score"].max()

                # Highlight verses that meet the threshold
                formatted_text = candidate["text"]
                for _, text in chapter_results.iterrows():
                    formatted_text = formatted_text.replace(
                        text["text"],
                        f'<span style="color:green">{text["text"]}</span>',
                    )

                # Do this only if
                # Create a dict to store the results
                biblegateway_url = urllib.parse.quote(
                    f"www.biblegateway.com/passage/?search={candidate['book']} {candidate['chapter']}&version=NIV"
                )
                result = {
                    "source": candidate["source"],
                    "url": biblegateway_url,
                    "score": max_score,
                    "formatted_text": formatted_text,
                    "raw_verse_results": chapter_results,
                }

                if max_score > 0:
                    all_results.append(result)

            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            # Trim down to specified n_results (can exceed due to multiple retriever approaches)
            all_results = all_results[:n_results]

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
                    if debug_mode:
                        st.table(result["raw_verse_results"])


if __name__ == "__main__":
    main()
