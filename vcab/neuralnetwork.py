import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, num_classes, dropout_prob=0.2):
        super(NeuralNetwork, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.output_layer = nn.Linear(layer_sizes[-1], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in layer_sizes])

    def forward(self, x):
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                x = self.relu(x)
                x = self.batch_norms[i](x)
                x = self.dropout(x)
        return self.output_layer(x)
