import torch

from .config import config
from .frontend import ChatDocFontend


def main():
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    frontend = ChatDocFontend()
    frontend.start()


if __name__ == "__main__":
    main()
