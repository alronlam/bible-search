from typing import List

import streamlit as st

from src.models import Chapter


class Reranker:
    def rerank(self, chapters: List[Chapter]) -> List[Chapter]:
        # TODO
        return chapters


# Rerankers applicable to SemanticRetriever results


class MaxVerseReranker(Reranker):
    def rerank(self, chapters: List[Chapter]) -> List[Chapter]:

        num_verses = [len(chapter.highlight_verses_df) for chapter in chapters]

        reranked_chapters = sorted(
            zip(chapters, num_verses), key=lambda x: x[1], reverse=True
        )
        reranked_chapters = [x[0] for x in reranked_chapters]

        return reranked_chapters
