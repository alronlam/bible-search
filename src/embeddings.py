import os
import traceback

import h5py
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingsManager:
    def __init__(self, model_name, bible_version, texts, embeddings_cache_dir) -> None:

        # Load embeddings model
        self.model = SentenceTransformer(model_name)

        # Load or generate embeddings baseed on the corpus
        sanitized_model_name = model_name.replace("\\", "-").replace("/", "-")
        self.cache_filename = f"{bible_version}_{sanitized_model_name}.h5"
        self.emb_cache_filepath = os.path.join(
            embeddings_cache_dir, self.cache_filename
        )

        # Load embeddings if it exists
        try:
            with h5py.File(self.emb_cache_filepath, "r") as h:
                self.embeddings = np.array(h["embeddings"])
        except Exception:
            traceback.print_exc()
            # If it doesn't, generate embeddings and save to a file
            logger.info(
                f"Generating embeddings and saving to {self.emb_cache_filepath}"
            )
            self.embeddings = self.model.encode(texts)
            with h5py.File(self.emb_cache_filepath, "w") as f:
                f.create_dataset("embeddings", data=self.embeddings)

        # Create a look-up dict to quickly retrieve embeddings of texts
        self.text_emb_dict = {}
        for text, embedding in zip(texts, self.embeddings):
            self.text_emb_dict[text] = embedding

        logger.info(
            f"Successfully loaded {model_name} embeddings for {bible_version} from {self.emb_cache_filepath}."
        )

    def get_embeddings(self, texts):
        embeddings = [self.text_emb_dict[text] for text in texts]
        return embeddings

    def __str__(self):
        return self.emb_cache_filepath


def score_semantic_similarity(query, texts_df):
    """Returns copy of text_df with semantic similarity scores."""
    pass
