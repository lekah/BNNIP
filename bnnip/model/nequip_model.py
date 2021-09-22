import numpy as np

from nequip.train.trainer import Trainer
from nequip.data import AtomicDataDict, AtomicData, Collater
from bnnip.model import AbstractModel, AbstractData
from bnnip.utils import zero_grad, parameters_gradients_to_vector


class NequipData(AbstractData):
    def __init__(self, dataset):
        # TODO: assert dataset
        self._dataset = dataset
        self._collater = Collater.for_dataset(dataset, exclude_keys=[])
        super(NequipData, self).__init__()

    def __len__(self):
        return len(self._dataset)

    def get_batch(self, indices=None):
        if indices is None:
            indices_ = range(len(self))
        elif isinstance(indices, int):
            indices_ = [indices]
        elif isinstance(indices, (tuple, list, set)):
            indices_ = [int(i) for i in indices]
        else:
            raise TypeError("Invalid type for indices")
        batch = self._collater([self._dataset.get(i) for i in indices_])
        return batch


class NequipModel(AbstractModel):
    def __init__(self, trainer):
        """
        pass
        """
        self._trainer = trainer
        # model needs to be set in training mood for gradient computation:
        self._trainer.model.train()

    def prepare_batch(self, batch):
        batch = batch.to(self._trainer.device)
        atomic_data_dict = AtomicData.to_AtomicDataDict(batch)
        if hasattr(self._trainer.model, "unscale"):
            # This means that self.model is RescaleOutputs
            # this will normalize the targets
            # in validation (eval mode), it does nothing
            # in train mode, if normalizes the targets
            atomic_data_dict = self._trainer.model.unscale(atomic_data_dict)
        return atomic_data_dict

    def train(self):
        self._trainer.model.train()

    def forward(self, atomic_data_dict):
        # computes only a forward pass
        # FOr now leave eval() commented out, since this ruins the results
        # a bit...
        # ~ self._trainer.model.eval()
        pred = self._trainer.model(atomic_data_dict)
        loss, loss_contrib = self._trainer.loss(pred=pred,
                                                ref=atomic_data_dict)
        return dict(loss=loss, pred=pred)

    def forward_backward(self, atomic_data_dict):
        zero_grad(self._trainer.model)
        # set model into training mode so that backward pass can be done
        # ~ self._trainer.model.train()
        pred = self._trainer.model(atomic_data_dict)
        loss, loss_contrib = self._trainer.loss(pred=pred,
                                                ref=atomic_data_dict)
        loss.backward()
        W, gradW = parameters_gradients_to_vector(
            self._trainer.model.parameters())
        return dict(loss=loss, W=W, gradW=gradW, pred=pred)

    def parameters(self):
        return self._trainer.model.parameters()

    def state_dict(self):
        return self._trainer.model.state_dict()
