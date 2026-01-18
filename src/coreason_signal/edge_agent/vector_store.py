from typing import List

import lancedb
from fastembed import TextEmbedding

from coreason_signal.schemas import SOPDocument
from coreason_signal.utils.logger import logger


class LocalVectorStore:
    """
    In-process vector store using LanceDB for Edge Agent RAG.
    """

    def __init__(self, db_path: str = "memory://", embedding_model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """
        Initialize the local vector store.

        Args:
            db_path: Path to the LanceDB database. Defaults to "memory://" for in-memory.
            embedding_model_name: Name of the embedding model to use with FastEmbed.
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self._db = lancedb.connect(db_path)
        self._table_name = "sops"
        self._embedding_model = TextEmbedding(model_name=self.embedding_model_name)

    def add_sops(self, sops: List[SOPDocument]) -> None:
        """
        Embed and store SOP documents.

        Args:
            sops: List of SOPDocument objects to add.
        """
        if not sops:
            return

        logger.info(f"Adding {len(sops)} SOPs to vector store")

        documents = [sop.content for sop in sops]
        # Generate embeddings
        embeddings = list(self._embedding_model.embed(documents))

        data = []
        for i, sop in enumerate(sops):
            item = sop.model_dump()
            item["vector"] = embeddings[i]
            data.append(item)

        try:
            table = self._db.open_table(self._table_name)
            table.add(data)
        except ValueError:
            # Table doesn't exist, create it
            self._db.create_table(self._table_name, data=data)

    def query(self, query_text: str, k: int = 3) -> List[SOPDocument]:
        """
        Semantic search for SOPs.

        Args:
            query_text: The query string.
            k: Number of results to return.

        Returns:
            List of SOPDocument objects matching the query.
        """
        try:
            table = self._db.open_table(self._table_name)
        except ValueError:
            logger.warning("Query attempted on empty vector store")
            return []

        # Embed the query
        query_embedding = list(self._embedding_model.embed([query_text]))[0]

        results = table.search(query_embedding).limit(k).to_list()

        sops = []
        for res in results:
            sop_data = {
                "id": res["id"],
                "title": res["title"],
                "content": res["content"],
                "metadata": res["metadata"],
                "associated_reflex": res.get("associated_reflex"),
            }
            sops.append(SOPDocument(**sop_data))

        return sops
