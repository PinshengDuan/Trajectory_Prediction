import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransAm(nn.Module):
    def __init__(self, feature_size, hidden_size=500, num_layers=2, dropout=0.1, lstm_hidden=250):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=feature_size, nhead=feature_size, dropout=dropout),
            num_layers=num_layers
        )
        self.lstm = nn.LSTM(feature_size, lstm_hidden, batch_first=True)
        self.decoder = nn.Linear(lstm_hidden, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.transformer_encoder(src, self.src_mask)
        src, _ = self.lstm(src)
        output = self.decoder(src)
        avg_predictions = torch.mean(output, dim=0)
        return avg_predictions

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
