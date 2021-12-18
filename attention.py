from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import torch
import torch.nn.functional as F

import math


class TransformerModel(nn.Module):
	def __init__(self, n_token: int, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float=0.5):
		super().__init__()
		self.model_type = "Transformer"
		self.pos_encoder = PositionEncoding(d_model, dropout)
		
		encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
		
		self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
		
		self.encoder = nn.Embedding(n_token, d_model)
		self.d_model = d_model
		self.decoder = nn.Linear(d_model, n_token)
		
		self.init_weights()
	
	def init_weights(self) -> None:
		init_range = 0.1
		self.encoder.weight.data.uniform_(-init_range, init_range)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-init_range, init_range)
	
	def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
		src = self.encoder(src) * math.sqrt(self.d_model)
		src = self.pos_encoder(src)
		
		output = self.transformer_encoder(src, src_mask)
		output = self.decoder(output)
		
		return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
	pass


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		
		pe = torch.zeros(max_len, 1, d_model)
		
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		
		self.register_buffer('pe', pe)
	
	def forward(self, x: Tensor) -> Tensor:
		"""
		args:
			x: Tensor, shape [seq_len, batch_size, embedding_dim]
		"""
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)
