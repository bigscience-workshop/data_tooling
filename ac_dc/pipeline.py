from . import filters, normalizers
from .config import BaseConfig


class BasePipeline:
    """
    Base class for all pipelines.
    """

    def __init__(self, config: BaseConfig):
        self.config = config

    def normalize(self, sentence: str) -> str:
        raise NotImplementedError

    def filter(self, sentence: str) -> str:
        raise NotImplementedError
