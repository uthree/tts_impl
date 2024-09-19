import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
    

class ConvReluNorm(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size=3,
            dropout_rate=0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, 1, dropout_rate//2)
        self.norm = nn.LayerNorm(output_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, self.dropout_rate)
        return x


class VariancePredictor(nn.Module):
    def __init__(
        self,
        input_channels=256,
        internal_channels=256,
        num_layers=1,
        kernel_size=3,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvReluNorm(input_channels, internal_channels, kernel_size, dropout_rate))
        for _ in range(num_layers):
            self.layers.append(ConvReluNorm(internal_channels, internal_channels, kernel_size, dropout_rate))
        self.layers.append(nn.Conv1d(internal_channels, 1, 1))

    def forward(self, x, x_mask):
        x = x * x_mask
        for layer in self.layers:
            x = layer(x) * x_mask
        return x


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2,3) * mask
    return path
    

class ScaledDotProductAttention(nn.Module):

    """ Scaled Dot-Product Attention """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):

    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_k is None:
            d_k = d_model

        if d_v is None:
            d_v = d_model

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
    


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
    

class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k=None, d_v=None, d_inner=None, kernel_size=5, dropout=0.1):
        if d_inner is None:
            d_inner = d_model * 4
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if mask is not None:
            enc_output = enc_output * mask

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output * mask

        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model 
        self.seq_len = seq_len 
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).detach()
        return
    

class VarianceAdapter(nn.Module):
    def __init__(
        self,
        d_model=256,
        channels=256,
        num_layers=1,
        kernel_size=3,
    ):
        super().__init__()
        self.duration_predictor = VarianceAdapter(d_model, channels, num_layers, kernel_size)
        self.pitch_predictor = VarianceAdapter(d_model, channels, num_layers, kernel_size)
        self.energy_predictor = VarianceAdapter(d_model, channels, num_layers, kernel_size)

        self.pitch_embedding = nn.Conv1d(1, d_model, 1)
        self.energy_embedding = nn.Conv1d(1, d_model, 1)
    
    def forward(self, x, x_mask, duration, pitch, energy):
        predicted_duration = self.duration_predictor(x, x_mask)

        attn = generate_path(duration, x_mask)
        x = torch.matmul(attn.squeeze(1), x.transpose(1, 2)).transpose(1, 2)

        x_mask = x_mask.transpose(1, 2)

        predicted_pitch = self.pitch_predictor(x, x_mask)
        predicted_energy = self.energy_predictor(x, x_mask)
        
        x = x + self.pitch_embedding(pitch) + self.energy_embedding(energy)
        
        return x, predicted_duration, predicted_pitch, predicted_energy
    

    def infer(self, x, x_mask, pitch=None, energy=None):
        pass