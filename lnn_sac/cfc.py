import torch
import torch.nn as nn


class CfCCell(nn.Module):
    """Closed-form Continuous-time (CfC) recurrent cell.

    Simplified fixed-timestep variant of Hasani et al.'s CfC (2022): instead of
    integrating an ODE, the hidden state is a data-dependent interpolation
    between two learned candidate states, with the interpolation gate itself a
    function of the input (a "liquid" time-constant, rather than a fixed one
    like a GRU's).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.candidate_a = nn.Linear(hidden_size, hidden_size)
        self.candidate_b = nn.Linear(hidden_size, hidden_size)
        self.time_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        z = self.backbone(torch.cat([x, h], dim=-1))
        a = torch.tanh(self.candidate_a(z))
        b = torch.tanh(self.candidate_b(z))
        gate = torch.sigmoid(self.time_gate(z) * dt)
        return gate * a + (1 - gate) * b
