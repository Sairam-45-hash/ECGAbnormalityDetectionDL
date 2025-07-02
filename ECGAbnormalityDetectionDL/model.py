import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation, padding=(kernel_size-1)*dilation)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class RhythmTCN_GRUAttNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.tcn = TemporalConvNet(1, [32, 64], kernel_size=3)

        # GRU input size = 1 (sequence length = 360)
        self.gru = nn.GRU(1, 64, batch_first=True, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(64 + 128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 360]
        tcn_out = self.tcn(x)                    # [B, 64, L]
        tcn_feat = torch.mean(tcn_out, dim=2)    # [B, 64]

        gru_input = x.squeeze(1).unsqueeze(-1)   # [B, 360, 1]
        gru_out, _ = self.gru(gru_input)         # [B, 360, 128]
        gru_feat = gru_out[:, -1, :]             # [B, 128]

        combined = torch.cat([tcn_feat, gru_feat], dim=1)  # [B, 192]

        # Save combined feature for attention extraction
        self.last_concat = combined

        attention_weights = self.attention(combined)       # [B, 1]
        attended = attention_weights * combined            # [B, 192]

        return self.classifier(attended)                   # [B, num_classes]
