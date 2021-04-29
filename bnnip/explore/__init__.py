


class Sampler(object):
    def __init__(self, model, mass):
        # TODO: checks
        if not isinstance(mass, float):
            raise TypeError("Mass has to be a float")
        self._mass = mass
        self.set_model(model)
    @property
    def model(self):
        return self._model
    def set_model(self, model):
        # TODO:checks
        self._model = model
