import torch
import torch.nn as nn
import torch.nn.functional as F
import nemo


class SOTA_TCN_Baseline(nn.Module):
    def __init__(self, num_channels=1, num_classes=5, Ft=64, Kt=11, Pt=0.2):
        super(SOTA_TCN_Baseline, self).__init__()

        # SOTA models typically use 32, 64, or 128 filters
        self.Ft = Ft
        self.Kt = Kt
        pt = Pt  # Standard dropout for TCNs

        # Initial Layer
        self.pad0 = nn.ConstantPad1d(padding=(Kt - 1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=num_channels, out_channels=Ft, kernel_size=Kt, bias=True)
        self.bn0 = nn.BatchNorm1d(Ft)
        self.act0 = nn.ReLU()

        # Block 1: Dilation 1
        self.block1 = self._make_tcn_block(Ft, Ft, dilation=1, p=pt)

        # Block 2: Dilation 2
        self.block2 = self._make_tcn_block(Ft, Ft, dilation=2, p=pt)

        # Block 3: Dilation 4
        self.block3 = self._make_tcn_block(Ft, Ft, dilation=4, p=pt)

        # Block 4: Dilation 8 (Ensures coverage for length 178)
        self.block4 = self._make_tcn_block(Ft, Ft, dilation=8, p=pt)

        # Final Linear layer
        self.linear = nn.Linear(in_features=Ft, out_features=num_classes)

    def _make_tcn_block(self, in_ch, out_ch, dilation, p):
        """Standard Residual TCN Block with Causal Padding"""
        return nn.ModuleDict(
            {
                "pad1": nn.ConstantPad1d(((self.Kt - 1) * dilation, 0), 0),
                "conv1": nn.Conv1d(in_ch, out_ch, self.Kt, dilation=dilation),
                "bn1": nn.BatchNorm1d(out_ch),
                "act1": nn.ReLU(),
                "drop1": nn.Dropout(p),
                "pad2": nn.ConstantPad1d(((self.Kt - 1) * dilation, 0), 0),
                "conv2": nn.Conv1d(out_ch, out_ch, self.Kt, dilation=dilation),
                "bn2": nn.BatchNorm1d(out_ch),
                "act2": nn.ReLU(),
                "drop2": nn.Dropout(p),
            }
        )

    def forward_block(self, x, block):
        res = x
        out = block["pad1"](x)
        out = block["conv1"](out)
        out = block["bn1"](out)
        out = block["act1"](out)
        out = block["drop1"](out)

        out = block["pad2"](out)
        out = block["conv2"](out)
        out = block["bn2"](out)
        out = block["act2"](out)
        out = block["drop2"](out)

        return out + res  # Residual addition

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Initial Feature Extraction
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)

        # Residual Blocks
        x = self.forward_block(x, self.block1)
        x = self.forward_block(x, self.block2)
        x = self.forward_block(x, self.block3)
        x = self.forward_block(x, self.block4)

        # SOTA Selection: Use the last timestep (most representative in causal nets)
        # Alternatively, use torch.mean(x, dim=2) for Global Average Pooling
        x = x[:, :, -1]

        return self.linear(x)
