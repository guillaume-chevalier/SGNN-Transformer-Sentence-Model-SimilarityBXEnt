# coding: utf-8

#
# # Self Governing Neural Network (SGNN): the Projection Layer
#
# This file is derived from the following file: 
#     https://github.com/guillaume-chevalier/SGNN-Self-Governing-Neural-Networks-Projection-Layer
#
# ## License:
# 
# BSD 3-Clause License
# 
# Copyright (c) 2018, Guillaume Chevalier
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.random_projection import SparseRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

from collections import Counter
from pprint import pprint


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
        string_ = string_.strip().replace(".", "." + sep).replace("?", "?" + sep).replace("!", "!" + sep).replace(";", ";" + sep).replace("\n", "\n" + sep)
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


class WordTokenizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        begin_of_word = "<"
        end_of_word = ">"
        out = [
            [
                begin_of_word + word + end_of_word
                for word in sentence.replace("//", " /").replace("/", " /").replace("-", " -").replace("  ", " ").split(" ")
                if not len(word) == 0
            ]
            for sentence in X
        ]
        return out


char_ngram_range = (1, 4)

char_term_frequency_params = {
    'char_term_frequency__analyzer': 'char',
    'char_term_frequency__lowercase': False,
    'char_term_frequency__ngram_range': char_ngram_range,
    'char_term_frequency__strip_accents': None,
    'char_term_frequency__min_df': 2,
    'char_term_frequency__max_df': 0.99,
    'char_term_frequency__max_features': int(1e7),
}


class CountVectorizer3D(CountVectorizer):

    def fit(self, X, y=None):
        X_flattened_2D = sum(X.copy(), [])
        super(CountVectorizer3D, self).fit_transform(X_flattened_2D, y)  # can't simply call "fit"
        return self

    def transform(self, X):
        return [
            super(CountVectorizer3D, self).transform(x_2D)
            for x_2D in X
        ]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


T = 80
d = 14

hashing_feature_union_params = {
    # T=80 projections for each of dimension d=14: 80 * 14 = 1120-dimensionnal word projections.
    **{'union__sparse_random_projection_hasher_{}__n_components'.format(t): d
       for t in range(T)
    },
    **{'union__sparse_random_projection_hasher_{}__dense_output'.format(t): False  # only AFTER hashing.
       for t in range(T)
    }
}

class FeatureUnion3D(FeatureUnion):

    def fit(self, X, y=None):
        X_flattened_2D = sp.vstack(X, format='csr')
        super(FeatureUnion3D, self).fit(X_flattened_2D, y)
        return self

    def transform(self, X):
        return [
            super(FeatureUnion3D, self).transform(x_2D).toarray()
            for x_2D in X
        ]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def get_sgnn_projection_pipeline(T=80, d=14, sgnn_training_data=None):
    params = dict()
    params.update(char_term_frequency_params)
    params.update(hashing_feature_union_params)

    pipeline = Pipeline([
        ("word_tokenizer", WordTokenizer()),
        ("char_term_frequency", CountVectorizer3D()),
        ('union', FeatureUnion3D([
            ('sparse_random_projection_hasher_{}'.format(t), SparseRandomProjection())
            for t in range(T)
        ]))
    ])
    pipeline.set_params(**params)

    if sgnn_training_data is None:
        with open("./data/How-to-Grow-Neat-Software-Architecture-out-of-Jupyter-Notebooks.md") as f:
            raw_data = f.read()
        sgnn_training_data = SentenceTokenizer().fit_transform(raw_data)

    pipeline.fit(sgnn_training_data)

    return pipeline
