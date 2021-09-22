#!/usr/bin/env python
import copy
import json
import os
import torch
import re

from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput
from nequip.utils import Config
from nequip.utils import dataset_from_config
from nequip.models import EnergyModel, ForceModel
from nequip.train.trainer import Trainer

from bnnip.explore.hmc import HMC
from bnnip.model.nequip_model import NequipModel, NequipData


DEFAULT_HMC_PARAMS = dict(
    mass=1.0, start_dt=1e-2,
    start_L=2500, temperature=1e-2,
    var_tot_threshold=1e-6, stability_criterion=1e-2,
    check_freq=-1,
    l_adaptation_factor=0.0, hd_batch_size=25,
    dt_reduction=0.5,
    verbosity=2)
    

def main_hmc(model_dir, model_parameters, model_config, hmc_parameters,
             nsteps, freq_save,  dataset=None, restart=False, starting_step=1,
             seed=None
            ):

    print("Start of HMC")
    print("model_dir: {}".format(model_dir))
    print("model_parameters will be read from: {}".format(model_parameters))
    print("model_config will be read from: {}".format(model_config))
    print("hmc_parameters will be read from: {}".format(hmc_parameters))
    print("dataset is: {}".format(dataset))
    print("nsteps: {}".format(nsteps))
    print("freq_save: {}".format(freq_save))
    print("restart: {}".format(restart))
    print("starting_step: {}".format(starting_step))
    print("seed: {}".format(seed))
    if model_dir is None:
        model_dir = '.'
    hmc_params = copy.deepcopy(DEFAULT_HMC_PARAMS)
    with open(hmc_parameters) as f:
        update_parameters = json.load(f)
        added_keywords = set(update_parameters.keys()).difference(set(hmc_params.keys()))
        if added_keywords:
            raise KeyError("Unknown keys {}".format(' '.join(added_keywords)))
        hmc_params.update(update_parameters)

    config = Config.from_file(model_config)

    if dataset:
        config.update(dict(dataset_file_name=dataset))

    dataset = dataset_from_config(config)

    trainer = Trainer(model=None, **dict(config))
    trainer.set_dataset(dataset)
    ((forces_std), (energies_mean, energies_std), (allowed_species, Z_count),
     ) = trainer.dataset_train.statistics(fields=[
         AtomicDataDict.FORCE_KEY, AtomicDataDict.TOTAL_ENERGY_KEY,
         AtomicDataDict.ATOMIC_NUMBERS_KEY, ],
        modes=["rms", "mean_std", "count"],)

    #energy_model = EnergyModel(**dict(config))
    #force_model = ForceModel(energy_model)
    force_model = ForceModel(**dict(config))
    core_model = RescaleOutput(
        model=force_model, scale_keys=[AtomicDataDict.FORCE_KEY,
                                       AtomicDataDict.TOTAL_ENERGY_KEY, ],
        scale_by=forces_std, shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY,
        shift_by=energies_mean,)
    # core_model = force_model
    if restart:
        if model_parameters is None:
            regex = re.compile('model-(?P<idx>\d+).pt')
            max_idx = -1
            for filename in os.listdir(model_dir):
                match = regex.match(filename)
                if match:
                    this_idx = int(match.group('idx'))
                    if this_idx > max_idx:
                        max_idx = this_idx
                        model_parameters = os.path.join(model_dir, match.group(0))
            if max_idx < 0:
                raise ValueError("restart but cannot find last model")
            starting_step = max_idx+1
            print("This is a restart!\nRestarting from {}\n"
                  "with starting_idx {}\n".format(model_parameters, starting_step))
        else:
            raise ValueError("model_parameters has to be left to None if restart")


    if model_parameters is None:
        # No model is provided, I will run a training
        print("Training model")
        trainer.model = core_model
        trainer.train()
        # ~ raise ValueError("model_parameters cannot be None if not restart")
    else:
        print("Loading model from {}".format(model_parameters))


        saved_state_dict = torch.load(model_parameters)
        core_model.load_state_dict(saved_state_dict)
        trainer.model = core_model
        trainer.init()

    # Creating the dataset:
    data = NequipData(dataset)

    # the nequipModel:
    model = NequipModel(trainer)
    print("HMC parameters:")
    for k, v in hmc_params.items():
        print('{:<20}  {}'.format(k,v))

    hmc_ = HMC(model=model, #copy.deepcopy(model), No need t deepcopy, model either trained OTF or loaded frmo file
               **hmc_params)
    hmc_.init_mc(data)
    if seed is not None:
        torch.random.manual_seed(seed)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    # Saving the initial model if it wasn't read in the same folder
    if not restart:
        hmc_.save_model(os.path.join(model_dir or '.', 'model-init.pt'))

    hmc_.run(nsteps, model_dir=model_dir, save_model_freq=freq_save,
            starting_step=starting_step,
            filename=os.path.join(model_dir, 'hmc.out'))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('model_config',
                        help='path to model configuration (yaml)')
    parser.add_argument('hmc_parameters',
                        help='path to hmc parameters (json)')
    parser.add_argument('model_dir',  help='save models  in this folder')
    parser.add_argument('-d', '--dataset',  help='use this dataset')
    parser.add_argument('-n', '--nsteps', type=int, default=100,
                        help='Run this many HMC steps')
    parser.add_argument('-m', '--model-parameters',
                        help='path to saved model parameters, optional.\n'
                        'If not provided and restart, will take last saved model.\n'
                        'If not provided and not restart, will train')
    parser.add_argument('-r', '--restart', action='store_true',
                        help='Run into restart, will figure which'
                        ' model to load and which starting step to choose'
                        ' from model_dir')
    parser.add_argument('-f', '--freq-save', type=int, default=1,
                        help='save model every N steps')
    parser.add_argument('-s', '--starting-step', type=int, default=1,
                        help='start at this step')

    parser.add_argument('--seed', type=int, help='initial seed')
    parsed = parser.parse_args()
    main_hmc(**vars(parsed))
