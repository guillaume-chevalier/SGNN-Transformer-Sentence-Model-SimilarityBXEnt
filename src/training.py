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


import time

import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score, f1_score, accuracy_score

from src.data.sgnn_projection_layer import *
from src.data.training_data import *
from src.model.loss import *
from src.model.save_load_model import save_model
from src.model.transformer import *


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model_trainer):
    d_model = model_trainer.sentence_projection_model.encoder.layers[0].size
    return NoamOpt(d_model, 2, 4000,
                   torch.optim.Adam(model_trainer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run_epoch(epoch, model_trainer, model_opt, data_batch_iterator, cuda_device_id):
    """
    Standard Training and Logging Function.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    mod_tokens = 0
    mod = 1  # TODO

    for i, (src, mask, category_per_sentence) in enumerate(data_batch_iterator):
        target_diagonal_block_matrix = categories_to_block_matrix(category_per_sentence)
        if cuda_device_id is not None:
            src = src.cuda(cuda_device_id)
            mask = mask.cuda(cuda_device_id)
            target_diagonal_block_matrix = target_diagonal_block_matrix.cuda(cuda_device_id)

        # forward.
        loss = model_trainer(src, mask, target_diagonal_block_matrix)
        total_loss += loss
        ntokens = (mask != 0.0).data.sum().item()
        total_tokens += ntokens
        mod_tokens += ntokens

        # backward.
        if model_opt is not None:
            loss.backward()
            model_opt.step()
            model_opt.optimizer.zero_grad()

        # log.
        if (i - 1) % mod == 0:
            elapsed = time.time() - start
            print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                  (epoch, i, loss / ntokens, mod_tokens / elapsed))
            start = time.time()
            mod_tokens = 0

    return total_loss / total_tokens


def train_model_on_data(
        max_epoch, train_iters_per_epoch, batch_size,
        epoch_model_name,
        preproc_sgnn_sklearn_pipeline=None,
        model_trainer=None,
        cuda_device_id=None,
        plot=True):
    # TODO: clean params.
    # batch_size = 128
    # train_iters_per_epoch = 24
    # test_iters_per_epoch = 2400
    # max_epoch = 10
    # cuda_device_id = 0  # None for CPU, 0 for first GPU, etc.

    # CUDA
    if cuda_device_id is not None:
        context = torch.cuda.device(cuda_device_id)
        context.__enter__()

    # Create model
    if preproc_sgnn_sklearn_pipeline is None:
        preproc_sgnn_sklearn_pipeline = get_sgnn_projection_pipeline()
    if model_trainer is None:
        sentence_projection_model = make_sentence_model()
        model_trainer = TrainerModel(sentence_projection_model)
    if cuda_device_id is not None:
        model_trainer = model_trainer.cuda(cuda_device_id)

    # Define some hand-crafted test data just for visualization purpose.:
    sentences_raw = (
        "This is a test. This is another test. "
        "I like bacon. I don't like bacon. "
        "My name is Guillaume. My family name is Chevalier. "
        "Programming can be used for solving complicated math problems. Let's use the Python language to write some scientific code. "
        "My family regrouped for Christmast. We met aunts and uncles. "
        "I like linux. I have an operating system. "
        "Have you ever been in the situation where you've got Jupyter notebooks (iPython notebooks) so huge that you were feeling stuck in your code?. Or even worse: have you ever found yourself duplicating your notebook to do changes, and then ending up with lots of badly named notebooks?. "
        "Either and in any ways. For every medium to big application. "
        "If you're working with notebooks, it is highly likely that you're doing research and development. If doing research and development, to keep your amazing-10x-working-speed-multiplier, it might be a good idea to skip unit tests. "
        "I hope you were satisfied by this reading. What would you do?."
    ).split(". ")  # each 2 sentence (pairs) above are similar, so we have 10 pairs as such:
    category_per_sentence = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    if plot:
        plot_a_result(
            category_per_sentence, cuda_device_id,
            preproc_sgnn_sklearn_pipeline, sentence_projection_model, sentences_raw)

    # Train model
    model_opt = get_std_opt(model_trainer)
    for epoch in range(max_epoch):
        model_trainer.train()
        run_epoch(
            epoch, model_trainer, model_opt,
            DataBatchIterator(preproc_sgnn_sklearn_pipeline, max_iters=train_iters_per_epoch, batch_size=batch_size),
            cuda_device_id
        )

        model_trainer.eval()
        run_epoch(
            epoch, model_trainer, model_opt,
            DataBatchIterator(preproc_sgnn_sklearn_pipeline, max_iters=1, batch_size=batch_size),
            cuda_device_id
        )
        this_epoch_model_name = epoch_model_name.format("{}", str(epoch).rjust(5, "0"))
        save_model(preproc_sgnn_sklearn_pipeline, sentence_projection_model, this_epoch_model_name)

    if plot:
        plot_a_result(
            category_per_sentence, cuda_device_id,
            preproc_sgnn_sklearn_pipeline, sentence_projection_model, sentences_raw)

    # CUDA
    if cuda_device_id is not None:
        context.__exit__()

    return preproc_sgnn_sklearn_pipeline, model_trainer


def plot_a_result(category_per_sentence, cuda_device_id, preproc_sgnn_sklearn_pipeline, sentence_projection_model,
                  sentences_raw):
    sentence_projection = preproc_sgnn_sklearn_pipeline.transform(sentences_raw)
    projected_words, mask = pad_right(sentence_projection)
    projected_words, mask = projected_words.cuda(cuda_device_id), mask.cuda(cuda_device_id)
    # Get the sentence representations:
    # (This is what can be reused as sentence projections to solve other NLP tasks with fine-tuning, or re-train everything jointly).
    sentence_projection = sentence_projection_model(projected_words, mask)
    # sentence_projection is of shape [sentences, projected_sentence_features] and is a CUDA torch array.
    # Score all sentences to each other to create a 2D symmetric similarity matrix:
    prediction = matching_network_self_attention(sentence_projection)
    clipped_prediction = ((prediction - prediction.mean() - prediction.std()) > 0)
    target_diagonal_block_matrix = categories_to_block_matrix(category_per_sentence)
    # Visualize.
    print("Matrices are symmetric, so on each border is a sentence dotted with annother one in "
          "similarity to get something that is almost like covariance of each sentences to each other. "
          "We should observe 2x2 activated blocks along the diagonal. The loss function is a binary "
          "cross-entropy on this sentence-to-sentence similarity grid we see. I seem to have invented a new "
          "similarity loss function but it probably already exists...")
    plt.imshow(prediction.data.cpu().numpy())
    plt.colorbar()
    plt.show()
    plt.imshow(clipped_prediction.data.cpu().numpy())
    plt.show()
    plt.imshow(target_diagonal_block_matrix.cpu().numpy())
    plt.show()

    print("Compute the 2D overlap in the matrix:")
    y_true = target_diagonal_block_matrix.cpu().numpy().flatten()
    y_pred = clipped_prediction.data.cpu().numpy().flatten()
    test_jaccard_score = jaccard_similarity_score(y_true, y_pred)
    test_f1_score = f1_score(y_true, y_pred)
    test_accuracy_score = accuracy_score(y_true, y_pred)
    print("test_jaccard_score:", test_jaccard_score)
    print("test_f1_score:", test_f1_score)
    print("test_accuracy_score:", test_accuracy_score)
