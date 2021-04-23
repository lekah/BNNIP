import copy

import numpy as np

import torch

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from bnnip.model import AbstractModel

class HamiltonianDynamics(object):
    def __init__(self, model, mass, dt, gamma=0.0, 
                 target_temperature=None, temp_control=None,
                 tau=None,):
        self._mass = mass
        self._dt = dt
        self._dt2 = dt**2
        self._tempcontrol = temp_control
        self._target_temperature = target_temperature
        if not isinstance(model, AbstractModel):
            raise TypeError("Model has to inherit frmo bnnip.model.AbstractModel")
        self.model = model

    def init_dynamics(self, batch):
        self._expl_data = self.model.prepare_batch(batch)

        # the weights are stored as a vector in _W:
        self._W = parameters_to_vector(self.model.parameters())
        # the number of degrees of freedom
        self._ndeg = self._W.shape[0]

        if self._tempcontrol is None:
            self._V  = self._W.new_zeros(size=self._W.shape)
        else:
            assert target_temperature is not None, "No temperature set"
            self._target_temperature = target_temperature
            self._V = self._create_velocities(self._ndeg)
            if self._tempcontrol == 'langevin':
                try:
                    self._gamma = float(gamma)
                    if self._gamma < 0 or self._gamma > 1:
                        raise
                except Exception:
                    raise ValueError("Gamma has to be a number between 0 and 1")
                self._langevin_factor = np.sqrt(
                        2*self._gamma * self._target_temperature / self._mass / self._dt)
            elif self._tempcontrol == 'andersen':
                try:
                    self._tau = float(tau)
                    if self._tau < 0:
                        raise
                except Exception:
                    raise ValueError("Tau has to be a positive number")
            else:
                raise ValueError("Unknown theremostat type {}".format(self._tempcontrol))
        self._calculate_kinetic_energy_temp()
        self._calculate_force()

    def _calculate_kinetic_energy_temp(self):
        """
        Calculates the kinetic energy of the system of weights, and
        the resulting temperatures, stores in self._kin, self._temps
        """
        self._kin = (0.5 * self._mass * self._V.pow(2).sum()).item()
        self._temp = 2*self._kin/self._ndeg

    def _create_velocities(self, size=None):
        if size is None:
            size = self._ndeg
        return torch.normal(mean=torch.zeros(size), 
                        std=np.sqrt(self._target_temperature/self._mass) * torch.ones(size))
    def _calculate_force(self):
        res = self.model.forward_backward(self._expl_data)
        self._F = - res['gradW']
        self._loss = res['loss'].item()
        
    def step(self):
        ret_val = (self._loss, self._kin, self._loss+self._kin, self._temp)
        acc = self._F / self._mass
        if self._tempcontrol == 'langevin':
            acc = acc - self._gamma * self._V + (
                self._langevin_factor * torch.normal(
                        mean=torch.zeros(self._ndeg), std=torch.ones(self._ndeg)) )
        self._W += (self._V * self._dt + 0.5 * acc * self._dt2 )
        vector_to_parameters(self._W, self.model.parameters())
        old_F = self._F.clone().detach()
        # calculating forces in new positions
        self._calculate_force()
        acc = (old_F + self._F) / self._mass
        self._V += 0.5 *  acc * self._dt
        if self._tempcontrol == 'andersen':
            rand = torch.rand(size=(self._ndeg,))
            msk = rand < self._tau/ self._dt
            self._V[msk] = self._create_velocities(msk.sum())
        self._calculate_kinetic_energy_temp()
        return ret_val

    def get_model(self):
        try:
            # The model needs to have the implementation to copy/deepcopy
            # for this to work:
            return copy.deepcopy(self.model)
        except:
            raise NotImplemented("Building a model has not been "
                "implemented")
            # TODO Build model old school
    def save_model(self, filename):
        torch.save(self.model.state_dict(),
                filename if filename.endswith('.pt') else filename+'.pt')
    def print_quantities(self,):
        print(self._loss, self._kin, self._loss + self._kin)

    def run(self, nsteps, save_model_freq=-1, print_freq=1,
            model_dir=None):
        """
        Runs nsteps steps of the dynamics. Saves intermediate models to model_dir
        every save_model_freq steps (-1 to never save).
        Saves every print_freq the loss, kinetic energy, total energy and temperature
        to file.
        """

        # ~ if print_file is None:
            # ~ print_file = 'run.log'
        # ~ with open(print_file, 'a') as f:
        for istep in range(nsteps):
            ret = self.step()
            if istep % print_freq == 0:
                loss, ekin, etot,  temp = ret
                print('{:<5} {:.6f} {:.6f} {:.6f} {:.6f}'.format(istep, *ret))
