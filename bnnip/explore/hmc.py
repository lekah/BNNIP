
import copy
import torch
from bnnip.explore import Sampler
from bnnip.explore.hamiltonian_dynamics import HamiltonianDynamics
from bnnip.model import AbstractData
from bnnip.utils import WelfordMeanM2


DT_REDUCTION = 0.5
MAX_TRIALS = 5
MAX_ATTEMPTS = 10
CHECK_FREQ = 10


class BadIntegrationError(Exception):
    pass


class HMC(Sampler):
    def __init__(self, model, mass=1.0, start_dt=1e-2, start_L=100,
                 temperature=1.0, hd_batch_size=100,
                 stability_criterion=1e-2, l_adaptation_factor=0.1):
        self._dt = float(start_dt)
        self._L = int(start_L)
        self._temperature = float(temperature)
        self._hd_batch_size = int(hd_batch_size)
        self._stability_criterion = float(stability_criterion)
        self._l_adaption_factor = float(l_adaptation_factor)
        super(HMC, self).__init__(model=model, mass=mass)

    def init_mc(self, data):
        if not isinstance(data, AbstractData):
            raise TypeError("Data has to be an instance of a subclass"
                            " of AbstractData")
        self._data = data
        full_batch = self._data.get_batch()
        self._full_atomic_data = self._model.prepare_batch(full_batch)
        self._full_loss = self.get_full_loss(self._model)

    def get_full_loss(self, model):
        return model.forward(self._full_atomic_data)['loss']

    def _check_variances(self, var_loss, var_tot):
        if var_tot/var_loss > self._stability_criterion:
            raise BadIntegrationError("Var loss:{:.4e} Var loss+kin:{:.4e}".format(
                var_loss, var_tot))

    def _perform_run(self, nsteps):
        retdict = {}
        hd = HamiltonianDynamics(copy.deepcopy(self._model), self._mass,
                                 dt=self._dt, target_temperature=None, temp_control=None,)
        # Setting the batch as a stochastic choice
        batch = self._data.get_rand_batch(self._hd_batch_size)
        hd.init_dynamics(batch)
        # setting velocities
        retdict['initial_kin'] = hd.set_boltzmann_velocities(self._temperature)[
            'kin']
        wf_loss = WelfordMeanM2()
        wf_tot = WelfordMeanM2()
        print_freq = int(nsteps/10)
        for istep in range(1, nsteps+1):
            loss, kin, tot, temp = hd.step()
            if istep % print_freq == 0:
                print('{:<5} {:.6f} {:.6f} {:.6f} {:.6f}'.format(istep,
                                                                 loss, kin, tot, temp))
            wf_loss.update(loss)
            wf_tot.update(tot)
            if istep % CHECK_FREQ == 0:
                _, var_loss, _ = wf_loss.estimate()
                _, var_tot, _ = wf_tot.estimate()
                self._check_variances(var_loss, var_tot)
        retdict['final_kin'] = hd.get_kinetic_energy()
        retdict['final_model'] = hd.model
        return retdict

    def _adapt_dt_L(self, factor):
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
        for iattempt in range(MAX_ATTEMPTS):
            # A trial is a tried HD run, which can be interrupted
            # due to bad integration or other problems
            for itrial in range(MAX_TRIALS):
                try:
                    # retdict containts starting kinetic energy, final kinetic
                    # energy and the final model
                    retdict = self._perform_run(self._L)
                    successful_hd = True
                    break
                except BadIntegrationError as e:
                    print("Caught BadIntegrationError: {}\n"
                          "Reducing dt {:.2e} -> {:.2e}".format(e,
                                                                self._dt, DT_REDUCTION*self._dt))
                    self._adapt_dt_L(DT_REDUCTION)
                    continue
                except ZeroDivisionError as e:
                    print("Non recoverable ZeroDivisionError, loss"
                          " is stable but loss+kin varies")
                    raise e

            if not successful_hd:
                raise RuntimeError("Could not get converged HD")
            # calculating full loss:
            final_loss = self.get_full_loss(retdict['final_model'])
            # p = min(1, exp(- (H(q*,p*) - H(q,p))))
            #   = min(1, exp(H(q,p) - H(q*,p*)))
            # ~ print(previous_loss, retdict['initial_kin'], final_loss + retdict['final_kin'])
            prop = torch.exp(((previous_loss + retdict['initial_kin']) -
                              (final_loss + retdict['final_kin'])) / self._temperature)
            r = torch.rand(size=(1,))[0]
            print('!Evaluating model!\n'
                  'Previous loss and kinetic: {}+{}={}\n'
                  'Current loss and kinetic: {}+{}={}\n'
                  'propability of acceptance is: {:.3f}\n'
                  'Random number drawn: {:.3f}'.format(
                      previous_loss.item(), retdict['initial_kin'],
                      previous_loss + retdict['initial_kin'],
                      final_loss.item(), retdict['final_kin'],
                      final_loss + retdict['final_kin'],
                      prop.item(), r.item()))
            if prop > r:
                successful_step = True
                self._full_loss = final_loss
                self.set_model(retdict['final_model'])
                print("New model accepted after {} attempts".format(iattempt+1))
                self._L = int(self._L*(1+self._l_adaption_factor))
                # TODO: increase _L?
                break
            else:
                # TODO reduce _L ?
                self._L = int(self._L*(1-self._l_adaption_factor))
                print("New model at attempt {} rejected".format(iattempt+1))
        if successful_step:
            pass
        else:
            raise RuntimeError("Max attempts surpassed, "
                               "could not obtain new model")
