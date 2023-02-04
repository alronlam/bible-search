import os
import time
from pathlib import Path

import streamlit as st

from src import bible_loader
from src.embeddings import EmbeddingsManager
from src.reranker import (
    CombinedScoreAndNumberReranker,
    MaxVerseReranker,
    Reranker,
    SemanticSimScoreReranker,
)
from src.retriever import Retriever, SemanticRetriever


def display_chapter(chapter):
    st.header(f"[{str(chapter)}]({chapter.get_biblegateway_url()})")
    chapter_text = chapter.get_formatted_text()
    st.markdown(chapter_text, unsafe_allow_html=True)
    # st.write(chapter.highlight_verses_df)


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

    retriever = SemanticRetriever(bible_df, embeddings_manager)
    # reranker = MaxVerseReranker()
    reranker = CombinedScoreAndNumberReranker()
    # reranker = SemanticSimScoreReranker()

    # DEBUG st.write(bible_df)

    # Get user input
    st.title("Verse Similarity Search")
    st.markdown(
        "Have you ever been stumped by a verse and wondered what other related things the Bible says about the topic? This tool was made just for that!"
    )
    query = st.text_input(
        "Put a verse's text here to find related verses...",
        "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.",
    )

    if query:
        with st.spinner("Searching..."):

            start = time.time()

            # Retrieve and re-rank
            candidate_chapters = retriever.retrieve(query, n=n_candidates)
            candidate_chapters = reranker.rerank(candidate_chapters)

            # Trim because candidates can be more than the desired results
            final_chapter_results = candidate_chapters[:n_results]

            # Display quick stats
            st.markdown(
                f"{len(final_chapter_results)} results found in {time.time()-start:.2f}s"
            )
            st.markdown("---")

            # Display results
            for chapter in final_chapter_results:
                display_chapter(chapter)
                st.markdown("---")


if __name__ == "__main__":
    main()
