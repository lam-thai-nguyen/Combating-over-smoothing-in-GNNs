import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        logits = self.convs[-1](x, edge_index)
        return logits
    
class SAGESkip(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        x = self.relu(self.input_proj(x))
        x0 = x

        for conv in self.convs[:-1]:
            x = conv(x, edge_index) + x0
            x = self.relu(x)
            x = self.dropout(x)

        logits = self.convs[-1](x, edge_index)
        return logits

class SAGELayer(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        all_x = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            all_x.append(x)

        x = torch.stack(all_x).sum(dim=0)
        logits = self.convs[-1](x, edge_index)
        return logits

class SAGESL(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        x = self.relu(self.input_proj(x))
        x0 = x

        all_x = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index) + x0
            x = self.relu(x)
            x = self.dropout(x)
            all_x.append(x)

        x = torch.stack(all_x).sum(dim=0)
        logits = self.convs[-1](x, edge_index)
        return logits
    