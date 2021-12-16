from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import torch
import torch.nn.functional as F


class TransformerModel(nn.Module):
