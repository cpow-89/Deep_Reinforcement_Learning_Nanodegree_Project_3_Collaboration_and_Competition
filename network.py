import torch.nn as nn


class ContinuousPolicyNetwork(nn.Module):
    """Simple Policy Network that maps states to continuous"""
    def __init__(self, config):
        super().__init__()
        self.network = self._create_network(config)

    @staticmethod
    def _create_network(config):
        return nn.Sequential(
            nn.Linear(config["observation_size"], config["fc_units"][0]),
            getattr(nn, config["activation_funcs"][0])(),
            nn.Linear(config["fc_units"][0], config["fc_units"][1]),
            getattr(nn, config["activation_funcs"][1])(),
            nn.Linear(config["fc_units"][1], config["action_size"]),
            getattr(nn, config["activation_funcs"][2])()
        )

    def forward(self, state):
        return self.network(state)
