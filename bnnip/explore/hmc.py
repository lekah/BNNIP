



# ~ _parameters = dict(
        # ~ mass=1.0, dt=2e-2,
    # ~ )


STABILITY_CRIT = 1e-5
class HMC(object):
    def __init__(self, model, mass, start_dt, start_N, temperature):
        pass


    def step(self):
        """
        Runs a single step of HMC:
        1. from current model, runs an HD simulations
        2. Evalutes the model at end of HD
        """
        pass
        
        
