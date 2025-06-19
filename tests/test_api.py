import unittest

from ragwebui.RAGRequestor import RAGRequestor
from ragwebui.__main__ import main


class TestRAGWebUI(unittest.TestCase):
    def test_main(self):
        main()

    def test_request(self):
        rag_requestor = RAGRequestor()
        rag_request = rag_requestor.rag_with_anchored_sources(message="Rituel de chevalier Kadosh")
        answer, html_sources = rag_request.to_html()
        print(answer)


if __name__ == "__main__":
    a = TestRAGWebUI()

    a.test_main()
    # a.test_request()
