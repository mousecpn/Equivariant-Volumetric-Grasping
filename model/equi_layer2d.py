"""Equivariant fully connected residual block used by induced decoders."""

from torch import nn
from escnn.nn import Linear, ReLU


class EquiResnetBlockFC(nn.Module):
    """Fully connected residual block over an ESCNN field type."""

    def __init__(self, in_type, out_type=None, hidden_type=None, normalize=False):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type if out_type is None else out_type
        self.hidden_type = self.out_type if hidden_type is None else hidden_type
        bias = not normalize
        self.fc_0 = Linear(self.in_type, self.hidden_type, bias=bias)
        self.fc_1 = Linear(self.hidden_type, self.out_type, bias=bias)
        self.actvn0 = ReLU(self.in_type, inplace=True)
        self.actvn1 = ReLU(self.hidden_type, inplace=True)
        self.shortcut = None if self.in_type == self.out_type else Linear(self.in_type, self.out_type, bias=False)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.fc_0(self.actvn0(x))
        return residual + self.fc_1(self.actvn1(x))

    def export(self):
        model = EquiResnetBlockFC(self.in_type, self.out_type, self.hidden_type)
        model.actvn0 = self.actvn0.export()
        model.actvn1 = self.actvn1.export()
        model.fc_0 = self.fc_0.export()
        model.fc_1 = self.fc_1.export()
        if self.shortcut is not None:
            model.shortcut = self.shortcut.export()
        return model
