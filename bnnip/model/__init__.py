from abc import ABCMeta, abstractmethod

class AbstractModel(metaclass=ABCMeta):
    @abstractmethod
    def prepare_batch(self, batch):
        """
        Prepare a batch
        """
        pass
    @abstractmethod
    def forward_backward(self, batch):
        """
        Runs forward and backward pass, returns dictionary with loss,
        gradients
        """
        pass
    @abstractmethod
    def parameters():
        """
        Returns iterable over parameters of the model
        """
        pass
