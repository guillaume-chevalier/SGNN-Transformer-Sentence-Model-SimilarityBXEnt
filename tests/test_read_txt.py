import os

import pytest
from sklearn.base import BaseEstimator

from src.data.config import UTF8_TXT_RAW_FILES
from src.data.read_txt import FilesReaderBinaryUTF8
from src.data.training_data import pad_right, SentenceTokenizer, DataBatchIterator
from src.data.sgnn_projection_layer import T, d, get_sgnn_projection_pipeline


def test_that_project_has_access_to_data_folder():
    folder_exists = os.path.exists(os.path.dirname(UTF8_TXT_RAW_FILES))
    assert folder_exists


def test_that_project_has_access_to_data_from_config_file():
    with FilesReaderBinaryUTF8(UTF8_TXT_RAW_FILES, pick_files_in_random_order=False, verbose=False) as f:
        raw_data = f.next_paragraph()
        print(raw_data)
        raw_data = f.next_paragraph()
        print(raw_data)
        raw_data = f.next_paragraph()
        print(raw_data)
        assert len(raw_data) != 0


list_of_sentences = [
    'How to Grow Neat Software Architecture out of Jupyter Notebooks.',
    'https://github.com/guillaume-chevalier/How-to-Grow-Neat-Software-Architecture-out-of-Jupyter-Notebooks'
]


def test_can_create_sgnn_preproc_pipeline():
    preproc_sgnn_sklearn_pipeline = get_sgnn_projection_pipeline(
        sgnn_training_data=list_of_sentences)  # TODO: more data

    assert isinstance(preproc_sgnn_sklearn_pipeline, BaseEstimator)


def test_sgnn_projection_layer_can_project():
    preproc_sgnn_sklearn_pipeline = get_sgnn_projection_pipeline(
        sgnn_training_data=list_of_sentences)
    projected_sentences = preproc_sgnn_sklearn_pipeline.transform(list_of_sentences)

    assert len(projected_sentences) != 0
    for proj in projected_sentences:
        assert len(proj.shape) == 2  # [words_of_sentence, features_of_word]
        assert proj.shape[-1] == T * d


def test_pad_right():
    raw_data = "This is a sentence. And this is another one. This test needs sentences as such."
    nb_sentences = 3
    max_sentence_length = 6
    preproc_sgnn_sklearn_pipeline = get_sgnn_projection_pipeline(
        sgnn_training_data=raw_data, T=T, d=d)
    list_of_sentences = SentenceTokenizer().fit_transform(raw_data)
    word_projections = preproc_sgnn_sklearn_pipeline.transform(list_of_sentences)

    word_projections, mask = pad_right(word_projections)

    assert word_projections.shape == (nb_sentences, max_sentence_length, T * d)
    assert mask.shape == (nb_sentences, 1, max_sentence_length)


def test_data_batch_iterator():
    sgnn_fit_data = "oiasdjf opiasd fjopias dfijop asdpoi fjakl;jswe fklzjsnd cvoiuasherf iuhans dfkjuans cvpiouha sdifjn asdkfjn"  # TODO.
    sgnn_projection_pipeline = get_sgnn_projection_pipeline(sgnn_training_data=sgnn_fit_data)
    max_iters = 3

    epoch = 0
    for src, mask in DataBatchIterator(sgnn_projection_pipeline, max_iters=max_iters):
        assert len(src.shape) == 3
        assert len(mask.shape) == 3
        epoch += 1
    assert epoch == 3


if __name__ == '__main__':
    pytest.main()
