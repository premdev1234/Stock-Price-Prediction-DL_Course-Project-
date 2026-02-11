import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """
    CNN + BiLSTM + Attention architecture
    Identical logic to final.py model
    """

    def __init__(self, input_dim):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, 32, kernel_size=3)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)

        self.attn = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attn(x), dim=1)
        x = (x * attn_weights).sum(dim=1)

        return self.fc(x).squeeze()


def build_model(input_dim):
    return HybridModel(input_dim)
