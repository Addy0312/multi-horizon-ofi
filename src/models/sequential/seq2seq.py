import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Seq2SeqAttentionModel(nn.Module):

    def __init__(self, input_dim: int, horizon_count: int, num_classes: int=3, enc_hidden: int=64, dec_hidden: int=64, enc_layers: int=2, attn_dim: int=64, dropout: float=0.3):
        super().__init__()
        self.horizon_count = horizon_count
        self.num_classes = num_classes
        self.encoder = nn.LSTM(input_dim, enc_hidden, num_layers=enc_layers, batch_first=True, dropout=dropout if enc_layers > 1 else 0.0, bidirectional=True)
        enc_out_dim = enc_hidden * 2
        self.attn_w = nn.Linear(enc_out_dim, attn_dim, bias=False)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.decoder = nn.LSTM(enc_out_dim, dec_hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dec_hidden, horizon_count * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        enc_out, _ = self.encoder(x)
        scores = self.attn_v(torch.tanh(self.attn_w(enc_out))).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (enc_out * weights).sum(dim=1, keepdim=True)
        _, (h, _) = self.decoder(context)
        z = self.drop(h[-1])
        return self.head(z).view(bsz, self.horizon_count, self.num_classes)

class Seq2SeqDeepLOBAttention(nn.Module):

    def __init__(self, input_dim: int, horizon_count: int, num_classes: int=3, hidden_dim: int=96, embed_dim: int=16):
        super().__init__()
        self.horizon_count = horizon_count
        self.num_classes = num_classes
        self.encoder_cnn = DeepLOBEncoder(conv_channels=64)
        self.encoder_lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.decoder_cell = nn.LSTMCell(hidden_dim + embed_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def _attend(self, enc_out: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(h_t).unsqueeze(2)
        score = torch.bmm(enc_out, q).squeeze(2)
        alpha = torch.softmax(score, dim=1)
        ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        enc_in = self.encoder_cnn(x)
        enc_out, (h_n, c_n) = self.encoder_lstm(enc_in)
        h_t = h_n[-1]
        c_t = c_n[-1]
        prev_cls = torch.ones(bsz, dtype=torch.long, device=x.device)
        logits_steps = []
        for _ in range(self.horizon_count):
            cls_vec = self.class_embed(prev_cls)
            ctx = self._attend(enc_out, h_t)
            dec_in = torch.cat([ctx, cls_vec], dim=1)
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            step_logits = self.classifier(torch.cat([h_t, ctx], dim=1))
            logits_steps.append(step_logits)
            prev_cls = step_logits.argmax(dim=1)
        return torch.stack(logits_steps, dim=1)

