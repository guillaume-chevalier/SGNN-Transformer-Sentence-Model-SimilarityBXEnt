import numpy as np
import pytest
import torch

from src.model.transformer import make_sentence_model, reduce_words_to_sentence_projection
from src.data.sgnn_projection_layer import T, d


def test_can_create_model():
    pytorch_model = make_sentence_model()

    assert isinstance(pytorch_model, torch.nn.Module)


def test_maxpool():
    nb_sentences = 3
    max_words_per_sentences = 6
    d_model = 512
    sentence_projection = torch.FloatTensor(np.random.random((nb_sentences, max_words_per_sentences, d_model)))

    print(sentence_projection.shape)  # torch.Size([3, 6, 512])
    pooled_across_words = reduce_words_to_sentence_projection(
        torch.nn.Linear(d_model, d_model),
        sentence_projection)
    print(pooled_across_words.shape)  # torch.Size([3, 512])

    assert pooled_across_words.shape == (nb_sentences, d_model)


def test_model_works():
    some_random_d_model = 2 ** 9
    five_sentences_of_twenty_words = torch.FloatTensor(np.random.random((5, 20, T * d)))
    five_sentences_of_twenty_words_mask = torch.FloatTensor(np.ones((5, 1, 20)))
    pytorch_model = make_sentence_model(d_model=some_random_d_model, T_sgnn=T, d_sgnn=d)

    output_before_match = pytorch_model(five_sentences_of_twenty_words, five_sentences_of_twenty_words_mask)

    print(output_before_match.shape)
    assert output_before_match.shape == (5, some_random_d_model)


if __name__ == '__main__':
    pytest.main()
