import torch.nn as nn
import torch.distributions as td

# Define the DenseHead class (adapted from LEXA)
class DenseHead(nn.Module):
    '''
    Dense head for the ensemble model.

    Predicts a normal distribution for each entry in the representation.
    Can use learned std or fixed std.
    '''

    def __init__(self, shape, layers, units, act=nn.ELU, std=1.0):
        super(DenseHead, self).__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._act = act
        self._std = std
        self.model_base = self.build_model_base()

        # define mean and std prediction heads
        self.mean = nn.Linear(self._units, self._shape)
        if self._std == 'learned':
            self.std = nn.Sequential(
                nn.Linear(self._units, self._shape),
                nn.Softplus()
            )

    def _build_model_base(self):
        model = [nn.Linear(self._shape, self._units)]
        model += [self._act()]
        for _ in range(self._layers - 1):
            model += [nn.Linear(self._units, self._units)]
            model += [self._act()]
        return nn.Sequential(*model)

    def forward(self):
        x = self.model_base(x)
        mean = self.mean(x)
        std = self.std(x) + 0.01 if self._std == 'learned' else self._std
        return td.independent.Independent(td.Normal(mean, std), len(self._shape))