from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import torch
import torch.nn.functional as F

from typing import Tuple

import math


class TransformerModel(nn.Module):
	def __init__(self, n_token: int, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float=0.5):
		super().__init__()
		self.model_type = "Transformer"
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		
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
	"""
	generates an upper-triangular matrix of -inf, with zeros on diag.
	"""
	return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


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


from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator


train_iter = WikiText2(split="train")
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
	# converts raw text into a flat tensor
	data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
	return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data: Tensor, bsz: int) -> Tensor:
	"""
	divides data into bsz separate sequences, removes extra elements that wouldn't cleanly fit.
	args:
		data: tensor, shape [N]
		bsz: int, batch size
	returns:
		tensor of shape [N // bsz, bsz]
	"""
	seq_len = data.size(0) // bsz
	data = data[:seq_len * bsz]
	data = data.view(bsz, seq_len).t().contiguous()
	return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 35

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
	"""
	args:
		source: tensor, shape [full_seq_len, batch_size]
		i: int
	returns:
		tuple (data, target), where data has shape [seq_len, batch_size] and
		target has shape [seq_len * batch_size]
	"""
	seq_len = min(bptt, len(source) - 1 - i)
	data = source[i:i + seq_len]
	target = source[i + 1:i + 1 + seq_len].reshape(-1)
	return data, target

n_tokens = len(vocab)	# size of vocabulary
emb_size = 200		# embedding dimension
d_hid = 200		# dimension of the feed-forward network model in nn.TransformerEncoder
n_layers = 2		# number of nn.TransformerEncoderLayer in nn.TransformerEncoder
n_head = 2		# number of heads in nn.MultiheadAttention
dropout = 0.2		# dropout probability
model = TransformerModel(n_tokens, emb_size, n_head, d_hid, n_layers, dropout).to(device)


import copy
import time


criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
	model.train()
	total_loss = 0.
	log_interval = 200
	start_time = time.time()
	src_mask = generate_square_subsequent_mask(bptt).to(device)
	
	num_batches = len(train_data) // bptt
	for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
		data, targets = get_batch(train_data, i)
		batch_size = data.size(0)
		if batch_size != bptt:  # last batch
			src_mask = src_mask[:batch_size, :batch_size]
		output = model(data, src_mask)
		loss = criterion(output.view(-1, n_tokens), targets)
		
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
		optimizer.step()
		
		total_loss += loss.item()
		if batch % log_interval == 0 and batch > 0:
			lr = scheduler.get_last_lr()[0]
			ms_per_batch = (time.time() - start_time) * 1000 / log_interval
			curr_loss = total_loss / log_interval
			ppl = math.exp(curr_loss)
			print(f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches| lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | loss {curr_loss:5.2f} | ppl {ppl:8.2f}")
			total_loss = 0
			start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
	model.eval()
	total_loss = 0.
	src_mask = generate_square_subsequent_mask(bptt).to(device)
	with torch.no_grad():
		for i in range(0, eval_data.size(0) - 1, bptt):
			data, targets = get_batch(eval_data, i)
			batch_size = data.size(0)
			if batch_size != bptt:
				src_mask = src_mask[:batch_size, :batch_size]
			output = model(data, src_mask)
			output_flat = output.view(-1, n_tokens)
			total_loss += batch_size * criterion(output_flat, targets).item()
	return total_loss / (len(eval_data) - 1)

best_val_loss = float("inf")
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
	epoch_start_time = time.time()
	train(model)
	val_loss = evaluate(model, val_data)
	val_ppl = match.exp(val_loss)
	elapsed = time.time() - epoch_start_time
	print("-" * 89)
	print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}")
	print("-" * 89)
	
	if val_loss < best_val_loss:
		best_val_loss = val_loss
		best_model = copy.deepcopy(model)
	
	scheduler.step()
