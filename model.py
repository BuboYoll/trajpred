import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        """
        hidden_size: the RNN/GRU encoder/decoder's hidden size.
        """
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, traj_de, traj_en):
        """
        traj_pred: Decoded seq (B, L_en, hidden_size)
        traj_past: Encoded seq (B, L_de, hidden_size)
        """

        if self.method == 'dot':
            wei = traj_de @ traj_en.transpose(-2, -1)  # (B, L_de, h) @ (B, h, L_en) -> (B, L_de, L_en)
        elif self.method == 'general':
            fc = self.attn(traj_en)  # (B, L_en, h)
            wei = traj_de @ fc.transpose(-2, -1)

        return F.softmax(wei, dim=-1)


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, loc_size, loc_emb_size, time_size, time_emb_size, hidden_size, attn_type='general',
                 rnn_type='GRU', dropout_p=0.5):
        """
        loc_size: in total how much locations in the dataset.
        tim_size: in total how much different timestamps in the dataset.
        hidden_size: RNN's hidden size
        """
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = loc_size
        self.loc_emb_size = loc_emb_size
        self.tim_size = time_size
        self.tim_emb_size = time_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        self.hidden_size = hidden_size
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.dropout_p = dropout_p

        input_size = self.loc_emb_size + self.tim_emb_size  # input features
        self.attn = Attn(self.attn_type, self.hidden_size)
        # self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size, bias=False)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, target_len):
        """
        loc: B, L
        tim: B, L
        """
        B, L = loc.shape  # Batch size, length
        loc_emb = self.emb_loc(loc)  # (B, L, loc_emb_size)
        tim_emb = self.emb_tim(tim)  # (B, L, tim_emb_size)
        x = torch.cat([loc_emb, tim_emb], dim=2)  # (B, L, input_size)
        x = self.dropout(x)

        if self.rnn_type == 'GRU':
            traj_en, h1 = self.rnn_encoder(x[:, :-target_len, :])  # (B, L_past, hidden_size)
            traj_de, h2 = self.rnn_decoder(x[:, -target_len:, :])  # (B, L_pred, hidden_size)

        elif self.rnn_type == 'LSTM':
            traj_en, (h1, c1) = self.rnn_encoder(x[:-target_len])
            traj_de, (h2, c2) = self.rnn_decoder(x[-target_len:])

        wei = self.attn(traj_de, traj_en)  # (B, L_pred, L_past)
        context = wei.bmm(traj_en)  # (B, L_pred, hidden size)
        out = torch.cat((traj_de, context), dim=-1)  # (B, L_pred, 2*hidden_size)
        out = self.dropout(out)

        y = self.fc_final(out)  # (B, L_pred, loc size)
        score = F.log_softmax(y, dim=-1)

        return score
