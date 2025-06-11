import unittest


from ragwebui.__main__ import main


class TestRAGWebUI(unittest.TestCase):
    def test_main(self):
        main()


if __name__ == "__main__":
    a = TestRAGWebUI()

    a.test_main()
