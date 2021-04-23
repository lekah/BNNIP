from nequip.data import Collater, AtomicDataDict

from nequip.nn import RescaleOutput
from nequip.utils import Config
from nequip.utils import dataset_from_config
from nequip.models import EnergyModel, ForceModel
from nequip.train.trainer import Trainer

from bnnip.explore.hamiltonian_dynamics import HamiltonianDynamics
from bnnip.model.nequip_model import NequipModel

import torch

def main(dt, mass, model_parameters, model_config, nsteps, freq_save,
        model_dir=None):
    config = Config.from_file(model_config)
    dataset = dataset_from_config(config)
    saved_state_dict = torch.load(model_parameters)
    trainer = Trainer(model=None, **dict(config))
    trainer.set_dataset(dataset)
    ((forces_std), (energies_mean, energies_std), (allowed_species, Z_count),
        ) = trainer.dataset_train.statistics(fields=[
            AtomicDataDict.FORCE_KEY, AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.ATOMIC_NUMBERS_KEY,],
        modes=["rms", "mean_std", "count"],)

    energy_model = EnergyModel(**dict(config))
    force_model = ForceModel(energy_model)
    core_model = RescaleOutput(
        model=force_model, scale_keys=[AtomicDataDict.FORCE_KEY, 
                                       AtomicDataDict.TOTAL_ENERGY_KEY,],
        scale_by=forces_std, shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY, 
        shift_by=energies_mean,)
    # core_model = force_model
    core_model.load_state_dict(saved_state_dict)
    trainer.model = core_model
    trainer.init()
    model = NequipModel(trainer)

    c = Collater.for_dataset(dataset, exclude_keys=[])
    batch = c([dataset.get(i) for i in range(10)])

    hd = HamiltonianDynamics(mass=mass, dt=dt, model=model)
    hd.init_dynamics(batch)

    hd.run(nsteps, save_model_freq=freq_save, model_dir=model_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-parameters', required=True, help='path to saved model parameters')
    parser.add_argument('-c', '--model-config', required=True, help='path to model configuration (yaml)')
    parser.add_argument('--mass',  help='mass', type=float, default=1.0)
    parser.add_argument('--dt',  help='time step', type=float, default=1e-2)

    parser.add_argument('-n', '--nsteps', type=int, help='number of steps')
    parser.add_argument('-f', '--freq-save', type=int, help='save model every N steps')
    parser.add_argument('--model-dir',  help='save model in this folder')
    parsed = parser.parse_args()
    main(**vars(parsed))
