import numpy as np
import pytest
import torch

from src.data.sgnn_projection_layer import get_sgnn_projection_pipeline
from src.data.training_data import pad_right
from src.model.transformer import reduce_words_to_sentence_projection, make_sentence_model
from src.model.loss import categories_to_block_matrix, TrainerModel


def test_categories_to_block_matrix():
    category_per_word = [0, 0, 1, 1, 2]

    target = categories_to_block_matrix(category_per_word)

    assert target.shape == (len(category_per_word), len(category_per_word))
    assert (target.data.numpy() == np.array([
        [1., 1., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [0., 0., 1., 1., 0.],
        [0., 0., 1., 1., 0.],
        [0., 0., 0., 0., 1.]]
    )).all()


def test_trainer_model():
    raw_data = "This is a sentence. And this is another one. This test needs sentences as such"
    category_per_word = [0, 1, 3]
    preproc_sgnn_pipeline = preproc_sgnn_sklearn_pipeline = get_sgnn_projection_pipeline(sgnn_training_data=raw_data)
    sentence_projection_model = make_sentence_model(d_ff=1024)
    list_of_sentences = raw_data.split(". ")
    projected_words = preproc_sgnn_pipeline.transform(list_of_sentences)
    projected_words, mask = pad_right(projected_words)

    model = TrainerModel(sentence_projection_model)

    model(projected_words, mask, category_per_word)


if __name__ == '__main__':
    pytest.main()
