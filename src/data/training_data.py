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

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.config import UTF8_TXT_RAW_FILES
from src.data.read_txt import FilesReaderBinaryUTF8


class SentenceTokenizer(BaseEstimator, TransformerMixin):
    # char lengths:
    MINIMUM_SENTENCE_LENGTH = 10
    MAXIMUM_SENTENCE_LENGTH = 200

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._split(X)

    def _split(self, string_):
        splitted_string = []

        sep = chr(29)  # special separator character to split sentences or phrases.
        string_ = string_.strip().replace(".", "." + sep).replace("?", "?" + sep).replace("!", "!" + sep).replace(
            ";", ";" + sep).replace("\n", "\n" + sep)
        for phrase in string_.split(sep):
            phrase = phrase.strip()

            while len(phrase) > SentenceTokenizer.MAXIMUM_SENTENCE_LENGTH:
                # clip too long sentences.
                sub_phrase = phrase[:SentenceTokenizer.MAXIMUM_SENTENCE_LENGTH].lstrip()
                splitted_string.append(sub_phrase)
                phrase = phrase[SentenceTokenizer.MAXIMUM_SENTENCE_LENGTH:].rstrip()

            if len(phrase) >= SentenceTokenizer.MINIMUM_SENTENCE_LENGTH:
                splitted_string.append(phrase)

        return splitted_string


def pad_right(projections, max_words_per_sentence=50):
    """
    :param projections: a list of 2D np.arrays of shape [nb_sentences, [sentence_length, d_model]].

    :return: A Torch tensor of shape [nb_sentences, max(sentence_length), d_model] concatenated and 0-padded to its right (far-end indices of value 0) on dim 1, and the mask of shape [nb_sentences, 1, max(sentence_length)] for that filled with zeros and ones.
    """
    sizes = [proj.shape[0] for proj in projections]
    max_dim_1 = min(max(sizes), max_words_per_sentence)

    padded_projections = torch.zeros(
        # [len(projections), projections[0].shape[-1], max_dim_1],
        [len(projections), max_dim_1, projections[0].shape[-1]],
        dtype=torch.float32)
    pad_mask = torch.zeros(
        [len(projections), 1, max_dim_1],
        dtype=torch.float32)

    for i, (size, proj) in enumerate(zip(sizes, projections)):
        # padded_projections[i, :, :size] = torch.from_numpy(proj).transpose(1, 0)
        if size > max_dim_1:
            proj = proj[:max_dim_1]
        padded_projections[i, :size, :] = torch.from_numpy(proj)
        pad_mask[i, :, :size] = 1.0
    # TODO: be sure that the new tensors doesn't memory-leak if always re-created.
    pad_mask.unsqueeze(-2)
    return padded_projections, pad_mask


class DataBatchIterator:

    def __init__(self, preproc_sgnn_sklearn_pipeline, max_iters, batch_size=25, max_words_per_sentence=50):
        self.batch_size = batch_size
        self.max_paragraph_sentences = int(batch_size / 2)
        self.max_words_per_sentence = max_words_per_sentence
        self.preproc_sgnn_sklearn_pipeline = preproc_sgnn_sklearn_pipeline
        self.max_iters = max_iters
        self.yielders = []

    def __iter__(self):
        for i in range(self.max_iters):
            yield self.get_batch()

    def get_batch(self):

        batch_word_projections = []  # self.next_yielder(0)
        category_per_sentence = []
        ii = 0
        while len(batch_word_projections) <= self.batch_size:
            new_batch = self.next_yielder(ii)
            new_batch_list = list(new_batch)
            batch_word_projections += new_batch_list
            category_per_sentence += [ii] * len(new_batch_list)
            ii += 1

        batch_word_projections = batch_word_projections[:self.batch_size]
        category_per_sentence = category_per_sentence[:self.batch_size]
        batch_word_projections, mask = pad_right(
            batch_word_projections, max_words_per_sentence=self.max_words_per_sentence)

        return batch_word_projections, mask, category_per_sentence

    def next_yielder(self, i):
        if len(self.yielders) <= i:
            self.yielders.append(self.yield_paragraphs().__iter__())
        return self.yielders[i].__next__()

    def yield_paragraphs(self):
        with FilesReaderBinaryUTF8(UTF8_TXT_RAW_FILES, pick_files_in_random_order=True, verbose=False) as f:
            # TODO: verbose...
            while True:
                raw_data = f.next_paragraph()
                sentences_data = SentenceTokenizer().fit_transform(raw_data)

                while len(sentences_data) > self.max_paragraph_sentences:
                    a = sentences_data[:self.max_paragraph_sentences]
                    yield self.preproc_sgnn_sklearn_pipeline.transform(a)
                    sentences_data = sentences_data[self.max_paragraph_sentences:]

                yield self.preproc_sgnn_sklearn_pipeline.transform(sentences_data)
