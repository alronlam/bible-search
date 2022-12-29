import os
from pathlib import Path

import streamlit as st

from src import bible_loader
from src.embeddings import EmbeddingsManager
from src.reranker import Reranker
from src.retriever import Retriever


def display_chapter(chapter):
    pass


def main():

    # Config
    ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
    DATA_DIR = ROOT_DIR / "data"

    n_results = 10
    n_candidates = n_results * 2
    metadata_csv = DATA_DIR / "key_english.csv"
    verses_csv = DATA_DIR / "NIV.csv"
    bible_version = "NIV"

    semantic_sim_model = "msmarco-distilbert-base-v4"

    # Initialize / Index
    bible_df = bible_loader.load_bible(metadata_csv, verses_csv)
    embeddings_manager = EmbeddingsManager(
        model_name=semantic_sim_model,
        bible_version=bible_version,
        embeddings_cache_dir=DATA_DIR,
        texts=bible_df["text"].tolist(),
    )

    retriever = Retriever()
    reranker = Reranker()

    # Get user input
    st.title("According to the Bible, ...")
    query = st.text_input("", "is it ok for a believer to continue in sin?")

    if query:
        with st.spinner("Searching..."):

            # Retrieve and re-rank
            candidate_chapters = retriever.retrieve(query, n=n_candidates)
            candidate_chapters = reranker.rerank(candidate_chapters)

            # Trim because candidates can be more than the desired results
            final_chapter_results = candidate_chapters[:n_results]

            # Display results
            for chapter in final_chapter_results:
                display_chapter(chapter)


if __name__ == "__main__":
    main()
