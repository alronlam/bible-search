import streamlit as st

from src import bible_loader
from src.embeddings import EmbeddingsManager
from src.reranker import Reranker
from src.retriever import Retriever


def display_chapter(chapter):
    pass


def main():

    # Initialize / Index
    bible_df = bible_loader.load_bible()
    embeddings_manager = EmbeddingsManager()

    retriever = Retriever()
    reranker = Reranker()

    # Config
    n_results = 10
    n_candidates = n_results * 2

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
