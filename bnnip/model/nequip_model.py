
from nequip.train.trainer import Trainer
from nequip.data import AtomicDataDict, AtomicData, Collater
from bnnip.model import AbstractModel
from bnnip.utils import zero_grad, parameters_gradients_to_vector


class NequipData(object):
    def __init__(self, dataset):
        # TODO: assert dataset
        self._dataset = dataset
        self._collater = Collater.for_dataset(dataset, exclude_keys=[])
        self._N = len(self._dataset)
        super(NequipData, self).__init__()
    def get_batch(self, indices=None):
        if indices is None:
            indices_ = range(self._N)
        elif isinstance(indices, int):
            indices_ = [indices]
        elif isinstance(indinces, (tuple, list, set)):
            indices_ = [int(i) for i in indices]
        else:
            raise TypeError("Invalid type for indices")
        batch = self._collater([self._dataset.get(i) for i in indices_])
        return batch
    def get_rand_batch(self, n):
        if not isinstance(n, int):

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
            atomic_data_dict  = self._trainer.model.unscale(atomic_data_dict)
        return atomic_data_dict

    def forward_backward(self, batch):
        zero_grad(self._trainer.model)
        pred = self._trainer.model(batch)
        loss, loss_contrib = self._trainer.loss(pred=pred, ref=batch)
        loss.backward()
        W, gradW = parameters_gradients_to_vector(self._trainer.model.parameters())
        return dict(loss=loss, W=W, gradW=gradW, pred=pred)
    def parameters(self):
        return self._trainer.model.parameters()

    def state_dict(self):
        return self._trainer.model.state_dict()
