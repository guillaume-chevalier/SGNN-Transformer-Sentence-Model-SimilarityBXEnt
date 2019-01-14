# coding: utf-8

#
# # Block Diagonal Matrix XENT Loss functions.
#
# ## License
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

import torch
from collections import Counter
import torch.nn.functional as F


def matching_network_self_attention(projected_sentences):
    """
    Attention from every query to every key.

    :param projected_sentences: tensor of shape torch.Size([sentence_count, d_cos_sim_features])
    :return: tensor of shape torch.Size([sentence_count, sentence_count])
    """
    projected_sentences_normalized = normalize_last_dim(projected_sentences)
    scores = torch.matmul(projected_sentences_normalized, projected_sentences_normalized.transpose(-1, -2))
    return scores


def normalize_last_dim(tensor, eps=1e-6):
    """
    Normalizes to a mean of 0 and a norm of 1.

    :param tensor: a ND tensor
    :param eps: epsilon to add not to divide by 0
    :return: the tensor with normalization along last dimension axis.
    """
    tensor = tensor - tensor.mean(-1, keepdim=True)
    tensor = tensor / (torch.norm(tensor, p=2, dim=-1, keepdim=True) + eps)
    return tensor


def categories_to_block_matrix(category_per_sentence):
    """
    Create a block matrix where each block entry is 1 and everything else is 0.
    Imagine this but where each block has value 1.0: https://i.ytimg.com/vi/a60r50XvtVo/maxresdefault.jpg

    Example input:
        category_per_sentence = [0, 0, 1, 1, 1, 2]  # categories' id must be sorted like this
    Example output for this input:
        tensor([[1., 1., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0.],
                [0., 0., 1., 1., 1., 0.],
                [0., 0., 1., 1., 1., 0.],
                [0., 0., 1., 1., 1., 0.],
                [0., 0., 0., 0., 0., 1.]])

    :param category_per_sentence: the list of categories id of each example. If counting occurences, we get the length of each block.
    :return: the corresponding block matrix, of shape [len(category_per_sentence), len(category_per_sentence)].
    """
    tmp = list(sorted(category_per_sentence))
    assert (tmp == category_per_sentence), "Categories must be in sorted order (increasing category id)"
    category_per_sentence = tmp
    size = len(category_per_sentence)
    block_matrix = torch.zeros((size, size))

    c = Counter(category_per_sentence)
    block_sizes = [val for key, val in sorted(list(c.items()))]
    a = 0
    for size in block_sizes:
        b = a + size
        block_matrix[a:b, a:b] = 1.0
        a = b

    return block_matrix


def loss_block_matrix_xent(prediction, target):
    prediction = prediction / 2.0 + 0.5
    prediction = prediction.clamp(0, 1)
    scaling_factor = 128
    return F.binary_cross_entropy(prediction, target, reduce=False).mean() * scaling_factor


class TrainerModel(torch.nn.Module):
    """Usage: TrainerModel(make_sentence_model())"""

    def __init__(self, sentence_projection_model):
        super(TrainerModel, self).__init__()
        self.sentence_projection_model = sentence_projection_model
        self.add_module("sentence_projection_model", sentence_projection_model)

    def __call__(self, x, mask, target_diagonal_block_matrix):
        sentence_projection = self.sentence_projection_model(x, mask)
        prediction = matching_network_self_attention(sentence_projection)
        loss = loss_block_matrix_xent(prediction, target_diagonal_block_matrix)
        return loss
