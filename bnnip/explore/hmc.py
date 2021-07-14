
import copy
import numpy as np
import torch
from bnnip.explore import Sampler
from bnnip.explore.hamiltonian_dynamics import HamiltonianDynamics
from bnnip.model import AbstractData
from bnnip.utils import WelfordMeanM2


MAX_TRIALS = 15
MAX_ATTEMPTS = 20


class BadIntegrationError(Exception):
    pass


class HMC(Sampler):
    _step_formatter = '{:>5d} {:.6f} {:.6f} {:.6f} {:.4e} {:.4e} {:d}'

    def __init__(self, model, mass=1.0, start_dt=1e-2, start_L=100,
                 temperature=1.0, hd_batch_size=100,
                 var_tot_threshold=1e-8, stability_criterion=1e-2,
                 check_freq=1, dt_reduction=0.5,
                 l_adaptation_factor=0.1, verbosity=0):
        self._dt = float(start_dt)
        self._L = int(start_L)
        self._temperature = float(temperature)
        self._hd_batch_size = int(hd_batch_size)
        self._stability_criterion = float(stability_criterion)
        self._var_tot_threshold = float(var_tot_threshold)
        self._check_freq = int(check_freq)
        self._l_adaption_factor = float(l_adaptation_factor)
        self._dt_reduction = float(dt_reduction)
        super(HMC, self).__init__(model=model, mass=mass,
                                  verbosity=verbosity)

    def init_mc(self, data):
        if not isinstance(data, AbstractData):
            raise TypeError("Data has to be an instance of a subclass"
                            " of AbstractData")
        self._data = data
        full_batch = self._data.get_batch()
        self._full_atomic_data = self._model.prepare_batch(full_batch)
        self._full_loss = self.get_full_loss(self._model)
        print("Starting run with loss at {:.6f}".format(self._full_loss))

    def get_full_loss(self, model):
        return model.forward(self._full_atomic_data)['loss']

    def _check_variances(self, var_loss, var_tot):
        if (var_tot/var_loss > self._stability_criterion and
                var_tot > self._var_tot_threshold):
            raise BadIntegrationError("Var loss:{:.4e} Var loss+kin:{:.4e}".format(
                var_loss, var_tot))

    def _perform_run(self, nsteps, dt):
        retdict = {}
        hd = HamiltonianDynamics(copy.deepcopy(self._model), self._mass,
                                 dt=dt, target_temperature=None,
                                 temp_control=None,)
        # Setting the batch as a stochastic choice
        batch = self._data.get_rand_batch(self._hd_batch_size)
        hd.init_dynamics(batch)
        # setting velocities
        retdict['initial_kin'] = hd.set_boltzmann_velocities(
            self._temperature)['kin']
        wf_loss = WelfordMeanM2()
        wf_tot = WelfordMeanM2()
        print_freq = int(nsteps/10)
        for istep in range(1, nsteps+1):
            loss, kin, tot, temp = hd.step()
            if self._verbosity > 1 and istep % print_freq == 0:
                print('{:<5} {:.6f} {:.6f} {:.6f} {:.6f}'.format(istep,
                                                                 loss, kin, tot, temp))
            wf_loss.update(loss)
            wf_tot.update(tot)
            if (self._check_freq > 0) and (istep % self._check_freq == 0):
                if np.isnan([loss, kin, tot, temp]).any():
                    raise BadIntegrationError("loss:{} kin:{} tot:{} temp:{}".format(
                                        loss, kin, tot, temp))

                mean_loss, var_loss = wf_loss.estimate()
                mean_tot, var_tot = wf_tot.estimate()
                self._check_variances(var_loss, var_tot)
        retdict['final_kin'] = hd.get_kinetic_energy()
        retdict['final_model'] = hd.model
        retdict['var_loss'] = wf_loss.estimate()[1]
        retdict['var_tot'] = wf_tot.estimate()[1]
        return retdict

    def _adapt_dt_L(self, factor):
        if factor ==1.0:
            return
        self._dt = factor*self._dt
        # also changing length of trajectory to compensate changed timestep
        self._L = int(self._L / factor)

    def step(self):
        """
        Runs a single step of HMC:
        1. from current model, runs an HD simulations
        2. Evalutes the model at end of HD
        """
        # The full loss was either calculated in init or during previous
        # step. This is the reference for acceptance/rejection
        previous_loss = copy.copy(self._full_loss)
        # Setting NVE style dynamics
        successful_hd = False
        # an attempt is a successful HD run, successful if configuration
        # is accept
        successful_step = False
        # Copying L and dt to local variable that can be adapted
        L = self._L
        dt = self._dt 
        for iattempt in range(MAX_ATTEMPTS):
            # A trial is a tried HD run, which can be interrupted
            # due to bad integration or other problems
            for itrial in range(MAX_TRIALS):
                try:
                    # retdict containts starting kinetic energy, final kinetic
                    # energy and the final model
                    retdict = self._perform_run(L, dt)
                    successful_hd = True
                    break
                except BadIntegrationError as e:
                    print("Caught BadIntegrationError: {}\n"
                          "Reducing dt {:.2e} -> "
                          "{:.2e}".format(e, dt, self._dt_reduction*dt))
                    #self._adapt_dt_L(self._dt_reduction) # Changing globally
                    #changing for this attemp
                    dt = self._dt_reduction*dt
                    L = int(L / self._dt_reduction)
                    continue
                except ZeroDivisionError as e:
                    print("Non recoverable ZeroDivisionError, loss"
                          " is stable but loss+kin varies")
                    raise e

            if not successful_hd:
                raise RuntimeError("Could not get converged HD")

            # calculating full loss:
            final_loss = self.get_full_loss(retdict['final_model'])

            # p = min(1, exp(- ((H(q*,p*) - H(q,p))) / T))
            #   = min(1, exp((H(q,p) - H(q*,p*))/T))
            prop = torch.exp(((previous_loss + retdict['initial_kin']) -
                              (final_loss + retdict['final_kin'])) /
                             self._temperature)
            if self._verbosity:
                print('!Evaluating model!\n'
                      'Previous loss and kinetic: {:.4f} + {:.4f} = {:.4f}\n'
                      'Current loss and kinetic: {:.4f} + {:.4f} = {:.4f}\n'
                      'propability of acceptance is: {:.3f}'.format(
                          previous_loss.item(), retdict['initial_kin'],
                          previous_loss + retdict['initial_kin'],
                          final_loss.item(), retdict['final_kin'],
                          final_loss + retdict['final_kin'],
                          prop.item() if prop <= 1 else 1))
            if prop > 1 or prop > torch.rand(size=(1,))[0]:
                successful_step = True
                self._full_loss = final_loss
                self.set_model(retdict['final_model'])
                if self._verbosity:
                    print("New model accepted at attempt {}".format(iattempt+1))
                if self._l_adaption_factor > 0.0:
                    self._L = int(self._L*(1+self._l_adaption_factor))
                break
            else:
                if self._l_adaption_factor > 0.0:
                    self._L = int(self._L*(1-self._l_adaption_factor))
                if self._verbosity:
                    print("New model rejected at attempt {}".format(iattempt+1))
        if successful_step:
            pass
        else:
            raise RuntimeError("Max attempts surpassed, "
                               "could not obtain new model")
        return (self._full_loss, retdict['final_kin'],
                final_loss + retdict['final_kin'],
                retdict['var_loss'], retdict['var_tot'], iattempt+1)
