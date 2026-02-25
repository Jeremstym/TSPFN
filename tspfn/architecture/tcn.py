import torch
import torch.nn as nn
import torch.nn.functional as F
import nemo

class Small_TCN(nn.Module):
    def __init__(self, num_channels=1, num_classes=5, seq_length=178, ft=11, kt=11, pt=0.3):
        super(Small_TCN, self).__init__()
        n_inputs = num_channels
        Kt = kt
        pt = pt
        Ft = ft
        classes = num_classes
        seq_length = seq_length  # Updated sequence length

        self.pad0 = nn.ConstantPad1d(padding = (Kt-1, 0), value = 0)
        self.conv0 = nn.Conv1d(in_channels = n_inputs, out_channels = n_inputs + 1, kernel_size = Kt, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features = n_inputs + 1)

        # Block 1: Dilation 1
        dilation = 1
        self.upsample = nn.Conv1d(in_channels = n_inputs + 1, out_channels = Ft, kernel_size = 1, bias=False)
        self.upsamplerelu = nn.ReLU()
        self.upsamplebn = nn.BatchNorm1d(num_features = Ft)
        self.pad1 = nn.ConstantPad1d(padding = ((Kt-1) * dilation, 0), value = 0)
        self.conv1 = nn.Conv1d(in_channels = n_inputs + 1, out_channels = Ft, kernel_size = Kt, dilation = dilation, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features = Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p = pt)
        self.pad2 = nn.ConstantPad1d(padding = ((Kt-1)*dilation, 0), value = 0)
        self.conv2 = nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features = Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p = pt)
        self.add1 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd1 = nn.ReLU()
        
        # Block 2: Dilation 2
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding = ((Kt-1) * dilation, 0), value = 0)
        self.conv3 = nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features = Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p = pt)
        self.pad4 = nn.ConstantPad1d(padding = ((Kt-1)*dilation, 0), value = 0)
        self.conv4 = nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features = Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p = pt)
        self.add2 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd2 = nn.ReLU()
        
        # Block 3: Dilation increased to 8 to cover 178 points
        dilation = 8 
        self.pad5 = nn.ConstantPad1d(padding = ((Kt-1) * dilation, 0), value = 0)
        self.conv5 = nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features = Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p = pt)
        self.pad6 = nn.ConstantPad1d(padding = ((Kt-1)*dilation, 0), value = 0)
        self.conv6 = nn.Conv1d(in_channels = Ft, out_channels = Ft, kernel_size = Kt, dilation = dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features = Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p = pt)
        self.add3 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd3 = nn.ReLU()

        # Last layer adjusted for sequence length 178
        # self.linear = nn.Linear(in_features = Ft * seq_length, out_features = classes, bias=False)
        self.linear = nn.Linear(in_features = Ft, out_features = classes, bias=False)      
        self.init_pact_bounds(6.0)

    def init_pact_bounds(self, alpha_val=6.0):
        """Initializes NEMO PACT alpha parameters using the correct attribute names."""
        for m in self.modules():
            # In newer NEMO, PACT_ReLU is often just PACT_Act
            if isinstance(m, (nemo.quant.pact.PACT_Act, nemo.quant.pact.PACT_IntegerAdd)):
                if hasattr(m, 'alpha'):
                    m.alpha.data.fill_(alpha_val)

    def forward(self, x):
        # Ensure input is [Batch, 1, 140]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Propagation
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)
        
        # TCN Blocks
        # Block 1
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)
        
        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)
        
        x = self.add1(x, res)
        x = self.reluadd1(x)
        
        # Block 2
        res = self.pad3(x)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = self.add2(x, res)
        x = self.reluadd2(x)
        
        # Block 3
        res = self.pad5(x)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = self.add3(x, res)
        x = self.reluadd3(x)
        
        x = torch.mean(x, dim=2)  # Global average pooling
        o = self.linear(x)
        return o


class SOTA_TCN_Baseline(nn.Module):
    def __init__(self, num_channels=1, num_classes=5, Ft=64, Kt=11, Pt=0.2):
        super(SOTA_TCN_Baseline, self).__init__()
        
        # SOTA models typically use 32, 64, or 128 filters
        self.Ft = Ft 
        self.Kt = Kt
        pt = Pt # Standard dropout for TCNs
        
        # Initial Layer
        self.pad0 = nn.ConstantPad1d(padding=(Kt-1, 0), value=0)
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
        return nn.ModuleDict({
            'pad1': nn.ConstantPad1d(((self.Kt-1) * dilation, 0), 0),
            'conv1': nn.Conv1d(in_ch, out_ch, self.Kt, dilation=dilation),
            'bn1': nn.BatchNorm1d(out_ch),
            'act1': nn.ReLU(),
            'drop1': nn.Dropout(p),
            'pad2': nn.ConstantPad1d(((self.Kt-1) * dilation, 0), 0),
            'conv2': nn.Conv1d(out_ch, out_ch, self.Kt, dilation=dilation),
            'bn2': nn.BatchNorm1d(out_ch),
            'act2': nn.ReLU(),
            'drop2': nn.Dropout(p)
        })

    def forward_block(self, x, block):
        res = x
        out = block['pad1'](x)
        out = block['conv1'](out)
        out = block['bn1'](out)
        out = block['act1'](out)
        out = block['drop1'](out)
        
        out = block['pad2'](out)
        out = block['conv2'](out)
        out = block['bn2'](out)
        out = block['act2'](out)
        out = block['drop2'](out)
        
        return out + res # Residual addition

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