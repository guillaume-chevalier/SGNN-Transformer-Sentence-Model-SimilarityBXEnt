import traceback

import pytest
from sklearn.base import BaseEstimator
from torch import nn

from src.data.sgnn_projection_layer import get_sgnn_projection_pipeline
from src.model.save_load_model import MY_MODEL_NAME, save_model, load_model, load_most_recent_model, delete_model
from src.model.transformer import make_sentence_model


def test_save_load():
    sgnn_fit_data = "oiasdjf opiasd fjopias dfijop asdpoi fjakl;jswe fklzjsnd cvoiuasherf iuhans dfkjuans cvpiouha sdifjn asdkfjn"  # TODO: ???.
    preproc_sgnn_sklearn_pipeline = get_sgnn_projection_pipeline(sgnn_training_data=sgnn_fit_data)
    sentence_projection_model = make_sentence_model()
    test_model_name = MY_MODEL_NAME + ".__unit_test_delete_this_file__"
    success = True
    try:

        # Test save.
        save_model(preproc_sgnn_sklearn_pipeline, sentence_projection_model, test_model_name)
        # Test load.
        preproc_sgnn_sklearn_pipeline, sentence_projection_model = load_model(test_model_name)
        preproc_sgnn_sklearn_pipeline2, sentence_projection_model2 = load_most_recent_model(test_model_name, None)

        assert isinstance(preproc_sgnn_sklearn_pipeline, BaseEstimator)
        assert isinstance(sentence_projection_model, nn.Module)
        assert isinstance(preproc_sgnn_sklearn_pipeline2, BaseEstimator)
        assert isinstance(sentence_projection_model2, nn.Module)
    except:
        traceback.print_exc()
        success = False
    finally:
        delete_model(test_model_name)
    assert success


if __name__ == '__main__':
    import sys

    sys.path.append("..")
    pytest.main([__file__])
