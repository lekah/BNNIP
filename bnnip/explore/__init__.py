import os
import torch
from abc import ABCMeta, abstractmethod


class Sampler(metaclass=ABCMeta):
    def __init__(self, model, mass, verbosity=0):
        # TODO: checks
        if not isinstance(mass, float):
            raise TypeError("Mass has to be a float")
        self._mass = mass
        self.set_model(model)
        self._verbosity = int(verbosity)

    @property
    def model(self):
        return self._model

    @abstractmethod
    def step(self):
        """
        Run a single step of the sampler
        """
        pass

    def set_model(self, model):
        # TODO:checks
        self._model = model

    def run(self, nsteps, save_model_freq=None, print_freq=1,
            model_dir=None, starting_step=1, save_file_freq=1,
            filename='hmc.out'):
        """
        Runs nsteps steps of the sampler, saving intermediate models to
        model_dir every save_model_freq steps (-1 or None to never save).
        Prints every print_freq sampled properties.
        """

        # ~ if print_file is None:
        # ~ print_file = 'run.log'
        # ~ with open(print_file, 'a') as f:

        if save_model_freq is not None and save_model_freq > 0:
            if model_dir is None:
                model_dir = '.'
            os.makedirs(model_dir, exist_ok=True)
            len_max_step = len(str(nsteps+starting_step))
            save_models = True
        else:
            save_models = False
        if filename is not None and save_file_freq > 0:
            save2file = True
            fil = open(filename, 'a')
        else:
            save2file = False
        for istep in range(starting_step, nsteps+starting_step):
            ret = self.step()
            if istep % print_freq == 0:
                print(self._step_formatter.format(istep, *ret))
            if save2file and istep % save_file_freq == 0:
                fil.write(self._step_formatter.format(istep, *ret)+'\n')
                fil.flush()
            if save_models and (istep % save_model_freq == 0):
                len_istep = len(str(istep))
                self.save_model(os.path.join(model_dir or '.', 'model-{}{}.pt'.format(
                    '0'*(len_max_step-len_istep), istep)))
        if save2file:
            fil.close()

    def get_model(self):
        try:
            # The model needs to have the implementation to copy/deepcopy
            # for this to work:
            return copy.deepcopy(self._model)
        except:
            raise NotImplemented("Building a model has not been "
                                 "implemented")
            # TODO Build model old school

    def save_model(self, filename):
        torch.save(self._model.state_dict(),
                   filename if filename.endswith('.pt') else filename+'.pt')
