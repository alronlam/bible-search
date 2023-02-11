import abc
from typing import List

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sparse_dot_topn import awesome_cossim_topn

from src.models import Chapter


class Retriever:
    @abc.abstractmethod
    def retrieve(self, query, n=10) -> List[Chapter]:
        pass


class SemanticRetriever:
    def __init__(
        self,
        bible_df,
        embeddings_manager,
        threshold=0.4,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    ):
        self.bible_df = bible_df
        self.embeddings_manager = embeddings_manager
        self.threshold = threshold
        self.cross_encoder_model = (
            CrossEncoder(cross_encoder_model) if cross_encoder_model else None
        )

        # 'cross-encoder/stsb-distilroberta-base'
        # cross-encoder/ms-marco-MiniLM-L-12-v2

    def retrieve(self, query, n=10) -> List[Chapter]:

        verse_candidates_df = self.semantic_search(
            query=query,
            texts=self.bible_df["text"].tolist(),
            embeddings_manager=self.embeddings_manager,
            n=n * 2,
            threshold=self.threshold,
        )

        if len(verse_candidates_df) == 0:
            return []

        if self.cross_encoder_model is not None:
            verse_candidates_df = self.cross_encode(
                query, verse_candidates_df["text"].tolist()
            )

        # TODO: revisit this logic as some verses can have the same exact text
        # For now, workaround is to drop duplicates
        verse_candidates_df.drop_duplicates(subset="text", inplace=True)

        # Join back verse metadata
        verse_candidates_df = pd.merge(
            verse_candidates_df, self.bible_df, how="left", on="text"
        )
        # DEBUG
        # st.write(verse_candidates_df)

        chapter_candidates = self.extract_chapters_from_verses(
            self.bible_df, verse_candidates_df
        )
        return chapter_candidates

    def cross_encode(self, query, texts):
        combinations = [[query, text] for text in texts]
        sim_scores = self.cross_encoder_model.predict(combinations)
        sim_scores = MinMaxScaler().fit_transform(sim_scores.reshape(-1, 1)).flatten()
        reranked_texts_scores = sorted(
            zip(texts, sim_scores), key=lambda x: x[1], reverse=True
        )
        df = pd.DataFrame(reranked_texts_scores, columns=["text", "score"])
        return df

    def semantic_search(self, query, texts, embeddings_manager, n=None, threshold=0):
        embeddings = embeddings_manager.get_embeddings(texts)
        query_embedding = embeddings_manager.get_embeddings([query])
        sim_scores = sklearn.metrics.pairwise.cosine_similarity(
            query_embedding, embeddings
        )[0]

        # Results is a list of tuples: [(text, score)]
        results = sorted(list(zip(texts, sim_scores)), key=lambda x: x[1], reverse=True)

        # Take top n only if specified
        if n:
            results = results[:n]

        # Apply a threshold to filter irrelevant results
        if threshold:
            results = [x for x in results if x[1] >= threshold]

        df = pd.DataFrame(results, columns=["text", "score"])

        return df

    def extract_chapters_from_verses(self, bible_df, verse_results_df) -> List[Chapter]:
        # Simple, naive assumption now is to just follow order of first appearance
        # I.e. The per-verse scores dictate the order
        # TODO: Revisit ranking

        # The goal here is to extract all the unique chapters based on the top verse results
        verse_results_df = verse_results_df.copy()
        verse_results_df["book_chapter"] = (
            verse_results_df["book"] + " " + verse_results_df["chapter"].astype(str)
        )
        unique_chapters = verse_results_df["book_chapter"].unique()

        bible_df = bible_df.copy()
        bible_df["book_chapter"] = (
            bible_df["book"] + " " + bible_df["chapter"].astype(str)
        )

        chapters = []
        for unique_chapter in unique_chapters:
            chapter_verses_df = bible_df[bible_df["book_chapter"] == unique_chapter]
            book = chapter_verses_df["book"].tolist()[0]
            chapter = chapter_verses_df["chapter"].tolist()[0]

            # Keep track of the matched verses as highlight verses
            highlight_verses_df = pd.merge(
                chapter_verses_df,
                verse_results_df[["text", "score", "book", "chapter"]],
                how="inner",
                on=["text", "book", "chapter"],
            )

            chapter = Chapter(
                book_name=book,
                chapter_num=chapter,
                verses_df=chapter_verses_df,
                highlight_verses_df=highlight_verses_df,
            )

            chapters.append(chapter)

        return chapters


class TfIdfRetriever(Retriever):
    def __init__(self, texts, preprocessors=[]) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
        self.preprocessors = preprocessors
        # TODO: pre-process the texts
        self.tfidf_vectors = self.vectorizer.fit_transform(texts)
        self.tfidf_vectors_transposed = self.tfidf_vectors.transpose()

    def search(self, query, n=10):
        query_tfidf_vector = self.vectorizer.transform([query])
        results = awesome_cossim_topn(
            query_tfidf_vector, self.tfidf_vectors_transposed, n, 0.01
        )
        return results
