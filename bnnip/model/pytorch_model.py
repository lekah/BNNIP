import numpy as np

from bnnip.model import AbstractModel, AbstractData
from bnnip.utils import zero_grad, parameters_gradients_to_vector
import torch

class SimpleData(AbstractData):
    def __init__(self, dataset):
        # TODO: assert torch array
        self._X, self._y = dataset
        super(SimpleData, self).__init__()

    def __len__(self):
        return len(self._X)

    def get_batch(self, indices=None):
        if indices is None:
            indices_ = torch.tensor(range(len(self)))
        elif isinstance(indices, int):
            indices_ = torch.tensor([indices])
        elif isinstance(indices, (tuple, list, set)):
            indices_ = torch.tensor([int(i) for i in indices])
        else:
            raise TypeError("Invalid type for indices")
        
        return self._X[indices_], self._y[indices_]
    @property
    def X(self):
        return self._X
    @property
    def y(self):
        return self._y


class SimpleModel(AbstractModel):
    def __init__(self, model, loss):
        """
        pass
        """
        self._model = model
        self._loss = loss
        self._model.train()

    def prepare_batch(self, batch):
        return SimpleData(batch)

    def train(self):
        self._model.train()

    def forward(self, batch):
        pred = self._model(batch.X)
        loss = self._loss(pred, batch.y)
        return dict(loss=loss, pred=pred)

    def forward_backward(self, batch):
        zero_grad(self._model)
        pred = self._model(batch.X)
        loss = self._loss(pred, batch.y)
        loss.backward()
        W, gradW = parameters_gradients_to_vector(
            self._model.parameters())
        return dict(loss=loss, W=W, gradW=gradW, pred=pred)

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()
