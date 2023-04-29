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


def config():
    n_results = st.sidebar.slider("Maximum Results?", 5, 30, 10)
    # bible_version = st.sidebar.selectbox("Bible Version", ["NIV", "ESV"]) # TODO
    bible_version = "NIV"
    new_testament = st.sidebar.checkbox("Search New Testament?", True)
    old_testament = st.sidebar.checkbox("Search Old Testament?", False)

    return n_results, new_testament, old_testament, bible_version


def main():

    st.set_page_config(page_title="Bible Search", layout="wide")

    n_results, new_testament, old_testament, bible_version = config()

    # Config
    ROOT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
    DATA_DIR = ROOT_DIR / "data"

    n_candidates = n_results * 2
    metadata_csv = DATA_DIR / "key_english.csv"
    verses_csv = DATA_DIR / f"{bible_version}.csv"

    semantic_sim_model = "msmarco-distilbert-base-v4"

    # Initialize / Index
    bible_df = bible_loader.load_bible(metadata_csv, verses_csv)
    embeddings_manager = EmbeddingsManager(
        model_name=semantic_sim_model,
        bible_version=bible_version,
        embeddings_cache_dir=DATA_DIR,
        texts=bible_df["text"].tolist(),
    )

    # Trim down search space if needed
    if not new_testament:
        bible_df = bible_df[bible_df["testament"] != "NT"]
    if not old_testament:
        bible_df = bible_df[bible_df["testament"] != "OT"]

    # Initialize retriever and reranker based on filtered texts
    retriever = SemanticRetriever(bible_df, embeddings_manager)
    reranker = CombinedScoreAndNumberReranker()
    # reranker = SemanticSimScoreReranker()
    # reranker = MaxVerseReranker()

    _, main_col, _ = st.columns([1, 2, 1])

    with main_col:

        # Get user input
        st.title("Verse Similarity Search")
        st.markdown(
            "- Have you ever been stumped by a verse and wondered what related things the Bible says about it?\n"
            "- Or you have a verse of interest and you simply want to find related ones?\n"
            "- Or you vaguely recall a verse's idea, but can't recall the exact text?\n"
            "This tool was made just for that!"
        )

        st.markdown("---")

        demo_query = st.selectbox(
            "Try some demo queries...",
            [
                "",
                "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.",
                "In the same way, faith by itself, if it is not accompanied by action, is dead.",
                "I tell you the truth, no one can enter the kingdom of God unless he is born of water and the Spirit.",
                "the Lord is patient with us, not wanting us to perish",
                "is it ok for believers to continue in sin?",
                "it is possible to resist every temptation",
                "heavenly rewards",
                "the old is gone, the new has come",
                "suffering for Christ",
                "rejoicing in trials",
                "Be careful of false prophets, wolves in sheep skin",
                "will there be marriage in heaven?",
            ],
            index=1,
        )

        query = st.text_area(
            "Or type a verse's text here to find similar verses",
            demo_query if demo_query.strip() else "",
        )

        clicked_search = st.button("Search", type="primary")

        if query or clicked_search:

            if len(bible_df) == 0:
                st.markdown(
                    "---\n:red[Please select at least one testament to search through (left hand side of the screen). :)]"
                )
            else:
                with st.spinner("Searching..."):

                    start = time.time()

                    # Retrieve and re-rank
                    candidate_chapters = retriever.retrieve(query, n=n_candidates)
                    candidate_chapters = reranker.rerank(candidate_chapters)

                    # Trim because candidates can be more than the desired results
                    final_chapter_results = candidate_chapters[:n_results]

                    # Display quick stats
                    st.markdown(
                        f"_{len(final_chapter_results)} results found in {time.time()-start:.2f}s_"
                    )
                    st.markdown("---")

                    # Display results
                    for chapter in final_chapter_results:
                        display_chapter(chapter)
                        st.markdown("---")


if __name__ == "__main__":
    main()
