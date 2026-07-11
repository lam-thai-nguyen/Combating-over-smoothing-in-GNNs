import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5):
        super().__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels, bias=True))
        for _ in range(num_layers-2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
        self.classifier = nn.Linear(hidden_channels, out_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        logits = self.classifier(x)
        return logits
    