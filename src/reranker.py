from typing import List

import numpy as np
import streamlit as st

from src.models import Chapter


class Reranker:
    def rerank(self, chapters: List[Chapter]) -> List[Chapter]:
        # TODO
        return chapters


# Rerankers applicable to SemanticRetriever results


def sort_chapters(chapters, scores):
    reranked_chapters = sorted(zip(chapters, scores), key=lambda x: x[1], reverse=True)
    reranked_chapters = [x[0] for x in reranked_chapters]
    return reranked_chapters


class CombinedScoreAndNumberReranker(Reranker):
    def __init__(self, num_verse_weight=0.3, semantic_sim_weight=0.7):
        self.num_verse_weight = num_verse_weight
        self.semantic_sim_weight = semantic_sim_weight

    def rerank(self, chapters: List[Chapter]) -> List[Chapter]:
        num_verse_score = compute_num_verse_scores(chapters)
        max_sem_sim_score = compute_sem_sim_scores(chapters)

        final_scores = (
            self.num_verse_weight * num_verse_score
            + self.semantic_sim_weight * max_sem_sim_score
        )
        return sort_chapters(chapters, final_scores)


class SemanticSimScoreReranker(Reranker):
    def rerank(self, chapters: List[Chapter]) -> List[Chapter]:
        sem_sim_scores = np.array(
            [chapter.highlight_verses_df["score"].max() for chapter in chapters]
        )
        return sort_chapters(chapters, sem_sim_scores)


class MaxVerseReranker(Reranker):
    def rerank(self, chapters: List[Chapter]) -> List[Chapter]:

        num_verses = [chapter.get_num_unique_highlight_verse() for chapter in chapters]

        return sort_chapters(chapters, num_verses)


def compute_num_verse_scores(chapters):
    num_verses = np.array(
        [chapter.get_num_unique_highlight_verse() for chapter in chapters]
    )
    max_verses = max(num_verses)
    num_verse_scores = num_verses / max_verses
    return num_verse_scores


def compute_sem_sim_scores(chapters):
    sem_sim_scores = np.array(
        [chapter.highlight_verses_df["score"].max() for chapter in chapters]
    )
    return sem_sim_scores
