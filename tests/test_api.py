import unittest

from ragwebui.config import config
from ragwebui.__main__ import main


class TestRAGWebUI(unittest.TestCase):
    def test_config(self):
        print(config)

    def test_main(self):
        main()

if __name__ == "__main__":
    a = TestRAGWebUI()

    # a.test_config()
    a.test_main()
    