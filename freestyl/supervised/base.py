from abc import ABC, abstractmethod
from freestyl.dataset.dataframe_wrapper import DataframeWrapper


class BaseSupervisedPipeline(ABC):
    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def fit(self,  data: DataframeWrapper, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, data: DataframeWrapper, *args, **kwargs):
        raise NotImplementedError()
