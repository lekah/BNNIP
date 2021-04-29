from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractData(metaclass=ABCMeta):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_batch(self, indices=None):
        pass

    def get_rand_batch(self, n):
        if not isinstance(n, int):
            raise TypeError("Number of batches has to be an integer")
        if n < 1:
            raise ValueError("N has to be positive")
        if n > len(self):
            raise ValueError("N ({}) larger than dataset ({})".format(
                n, len(self)))
        indices = np.random.choice(np.arange(len(self)), size=n,
                                   replace=False).tolist()
        return self.get_batch(indices)


class AbstractModel(metaclass=ABCMeta):
    @abstractmethod
    def prepare_batch(self, batch):
        """
        Prepare a batch
        """
        pass

    @abstractmethod
    def forward(self, batch):
        """
        Runs forward , returns dictionary with loss
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
