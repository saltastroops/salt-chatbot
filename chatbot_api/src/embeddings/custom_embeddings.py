import os

from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions


class MyEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A custom embedding function wrapper for embedding documents and queries
    using a Sentence Transformer model.
    """
    def __init__(self, model_name: str = None, *args, **kwargs):
        """
        Initialize the embedding function with compatibility for the base class.

        :param model_name: Name of the model to use for embeddings. Defaults to the value from the environment variable.
        """
        super().__init__(*args, **kwargs)  # Initialize the base class

        # Determine model name
        self.model_name = model_name or os.getenv("EMBEDDING_FUNC_MODEL")
        if not self.model_name:
            raise ValueError("EMBEDDING_FUNC_MODEL environment variable is not set, and no model name was provided.")

        # Initialize the SentenceTransformerEmbeddingFunction
        try:
            self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize SentenceTransformerEmbeddingFunction with model '{self.model_name}': {e}")

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embed a batch of documents.

        :param input: A list of documents to embed.
        :return: Embeddings for the provided documents.
        """
        return self.sentence_transformer_ef(input)

    def embed_documents(self, input: Documents) -> Embeddings:
        """
        Embed multiple documents.

        :param input: A list of documents to embed.
        :return: Embeddings for the provided documents.
        """
        return self.sentence_transformer_ef(input)

    def embed_query(self, input: str) -> Embeddings:
        """
        Embed a single query.

        :param input: A single query string to embed.
        :return: Embeddings for the query.
        """
        return self.sentence_transformer_ef([input])
