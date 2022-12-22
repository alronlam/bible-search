from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


class Retriever:
    def __init__(self, texts, preprocessors=[]) -> None:
        self.vectorizer = TfidfVectorizer()
        self.preprocessors = preprocessors
        # TODO: pre-process the texts
        self.tfidf_vectors = self.vectorizer.fit_transform(texts)
        self.tfidf_vectors_transposed = self.tfidf_vectors.transpose()

    def search(self, query, n_results=10):
        query_tfidf_vector = self.vectorizer.transform([query])
        results = awesome_cossim_topn(
            query_tfidf_vector, self.tfidf_vectors_transposed, n_results, 0.01
        )
        return results
