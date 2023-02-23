import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, encoder_cfg, input_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = encoder_cfg.get("hidden_size")
        self.device = device

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, x):
        embedding = self.embedding(x)
        _, hidden = self.gru(embedding)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, decoder_cfg, output_size, device):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = decoder_cfg.get("hidden_size")
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x, hidden):
        embedding = self.embedding(x)
        embedding = self.relu(embedding)

        output, hidden = self.gru(embedding, hidden)
        output = self.out(output)
        output = self.softmax(output)
        
        return output, hidden
