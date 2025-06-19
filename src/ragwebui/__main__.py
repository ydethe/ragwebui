import torch

from .config import config
from .frontend import start_frontend
from .RAGRequestor import RAGRequestor


def main():
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    rag_requestor = RAGRequestor()
    start_frontend(rag_requestor)


if __name__ == "__main__":
    main()
