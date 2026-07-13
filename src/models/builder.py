
import torch.nn as nn
from .sequential.cnn_lstm import HybridCNNInceptionLSTM
from .transformer.dilated import DilatedMaskedTransformer
from .sequential.seq2seq import Seq2SeqAttentionModel, Seq2SeqDeepLOBAttention
from .autoencoder.lstm_ae import LSTMAutoencoder

def build_deep_model(arch: str, input_dim: int, horizon_count: int, num_classes: int=3) -> nn.Module:
    if arch == 'dilated_transformer':
        return DilatedMaskedTransformer(input_dim=input_dim, horizon_count=horizon_count, num_classes=num_classes, d_model=96, n_heads=4, n_layers=2, dropout=0.15)
    if arch == 'hybrid_cnn_inception_lstm':
        return HybridCNNInceptionLSTM(input_dim=input_dim, horizon_count=horizon_count, num_classes=num_classes, channels=96, lstm_hidden=96, dropout=0.2)
    if arch == 'seq2seq_attn':
        return Seq2SeqDeepLOBAttention(input_dim=input_dim, horizon_count=horizon_count, num_classes=num_classes, hidden_dim=96, embed_dim=16)
    raise ValueError(f'Unknown architecture: {arch}')

