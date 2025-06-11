import unittest

from qdrant_client import QdrantClient

from ragwebui.config import config
from ragwebui.__main__ import main


class TestRAGWebUI(unittest.TestCase):
    def test_config(self):
        qdrant = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
            https=config.QDRANT_HTTPS,
        )
        hits = qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=[0.0] * 384,
            limit=config.QDRANT_QUERY_LIMIT,
            with_payload=True,
        ).points
        print(hits[0])

    def test_main(self):
        main()


if __name__ == "__main__":
    a = TestRAGWebUI()

    a.test_config()
    # a.test_main()
