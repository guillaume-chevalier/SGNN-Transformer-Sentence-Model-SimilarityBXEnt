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
    sentence_projection = torch.from_numpy(np.random.random((nb_sentences, max_words_per_sentences, d_model))).float()

    print(sentence_projection.shape)  # torch.Size([3, 6, 512])
    pooled_across_words = reduce_words_to_sentence_projection(
        torch.nn.Linear(d_model, d_model),
        sentence_projection)
    print(pooled_across_words.shape)  # torch.Size([3, 512])

    assert pooled_across_words.shape == (nb_sentences, d_model)


def test_model_works():
    some_random_d_model = 2 ** 9
    five_sentences_of_twenty_words = torch.from_numpy(np.random.random((5, 20, T * d))).float()
    five_sentences_of_twenty_words_mask = torch.from_numpy(np.ones((5, 1, 20))).float()
    pytorch_model = make_sentence_model(d_model=some_random_d_model, T_sgnn=T, d_sgnn=d)

    output_before_match = pytorch_model(five_sentences_of_twenty_words, five_sentences_of_twenty_words_mask)

    print(output_before_match.shape)
    assert output_before_match.shape == (5, some_random_d_model)


def test_model_works_on_gpu():
    device_id = 0
    with torch.cuda.device(device_id) as cuda:
        some_random_d_model = 2 ** 9
        five_sentences_of_twenty_words = torch.from_numpy(np.random.random((5, 20, T * d))).float()
        five_sentences_of_twenty_words_mask = torch.from_numpy(np.ones((5, 1, 20))).float()
        pytorch_model = make_sentence_model(d_model=some_random_d_model, T_sgnn=T, d_sgnn=d)

        five_sentences_of_twenty_words = five_sentences_of_twenty_words.cuda(device_id)
        five_sentences_of_twenty_words_mask = five_sentences_of_twenty_words_mask.cuda(device_id)
        print(type(five_sentences_of_twenty_words), type(five_sentences_of_twenty_words_mask))
        print(five_sentences_of_twenty_words.is_cuda, five_sentences_of_twenty_words_mask.is_cuda)
        pytorch_model = pytorch_model.cuda(device_id)
        output_before_match = pytorch_model(five_sentences_of_twenty_words, five_sentences_of_twenty_words_mask)

        assert output_before_match.shape == (5, some_random_d_model)
        print(type(output_before_match))
        print(output_before_match.is_cuda, output_before_match.get_device())
        assert output_before_match.is_cuda
        assert five_sentences_of_twenty_words.is_cuda
        assert five_sentences_of_twenty_words_mask.is_cuda


if __name__ == '__main__':
    pytest.main()
