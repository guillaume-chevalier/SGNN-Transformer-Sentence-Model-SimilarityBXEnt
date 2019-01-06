# coding: utf-8

#
# # The annotated transformer
#
# This file may contain code from: https://github.com/harvardnlp/annotated-transformer
#
# ## Original license:
#
#     MIT License
#
#     Copyright (c) 2018 Alexander Rush
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
#
# ## New license for the modified work:
#
#     BSD 3-Clause License
#
#     Copyright (c) 2018, Guillaume Chevalier
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#
#     * Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ProjectEncode(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, sgnn_projector, encoder):
        super(ProjectEncode, self).__init__()
        self.sgnn_projector = sgnn_projector
        self.encoder = encoder
        self.add_module("sgnn_projector", sgnn_projector)
        self.add_module("encoder", encoder)

    def forward(self, x, mask):
        "Take in and process sequences, using word projector and encoder."
        # x = x  # .cuda(0)
        # mask = mask  # .cuda(0)
        x_projected = self.sgnn_projector(x)
        return self.encoder(x_projected, mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N_encoder_layers, d_model):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N_encoder_layers)
        self.norm = LayerNorm(layer.size)
        self.linear = nn.Linear(d_model, d_model)
        self.add_module("layers", self.layers)
        self.add_module("norm", self.norm)
        self.add_module("linear", self.linear)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return reduce_words_to_sentence_projection(self.linear, x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def reduce_words_to_sentence_projection(linear, sentence_projection):
    """
    Reduce many words to one sentence-level projection, using max pooling.

    :param sentence_projection: torch.Size([nb_sentences, max_words_per_sentences, d_model])
    :return: # torch.Size([nb_sentences, d_model])
    """
    sentence_projection_t = sentence_projection.transpose(2, 1)
    pooled_across_words = F.max_pool1d(
        sentence_projection_t, kernel_size=sentence_projection_t.size()[-1]).squeeze(-1)
    projected_sentences = linear(pooled_across_words)
    return projected_sentences


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 2)
        self.size = size
        self.add_module("self_attn", self.self_attn)
        self.add_module("feed_forward", self.feed_forward)
        self.add_module("sublayer", self.sublayer)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.add_module("norm", self.norm)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + sublayer(self.norm(x))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = int(h)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.add_module("linears", self.linears)

    def forward(self, query, key, value, mask):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask = mask.cuda(0)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.add_module("w_1", self.w_1)
        self.add_module("w_2", self.w_2)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .cuda(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


class PositionwiseFeedForwardForWordProjections(nn.Module):
    def __init__(self, d_model, T_sgnn, d_sgnn):
        super(PositionwiseFeedForwardForWordProjections, self).__init__()
        self.d_model = d_model
        self.w = nn.Linear(T_sgnn * d_sgnn, d_model)  # .cuda(0)
        self.add_module("w", self.w)

    def forward(self, x):
        # x = x.cuda(0)
        # self.w = self.w.cuda(0)
        h = self.w(x)
        lrelu = F.relu(h) + h / 5.0
        return lrelu


def make_sentence_model(N_encoder_layers=6, d_model=512, d_ff=2048, h=8, T_sgnn=80, d_sgnn=14):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    position = PositionalEncoding(d_model)

    model = ProjectEncode(
        nn.Sequential(PositionwiseFeedForwardForWordProjections(d_model, T_sgnn, d_sgnn), c(position)),
        Encoder(EncoderLayer(d_model, c(attn), c(ff)), N_encoder_layers, d_model)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
