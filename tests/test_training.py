import traceback

import pytest

from src.model.save_load_model import delete_model, MY_MODEL_NAME
from src.training import train_model_on_data


def test_can_train_one_epoch_of_one_step():
    batch_size = 10
    train_iters_per_epoch = 1
    max_epoch = 1
    cuda_device_id = None  # None for CPU, 0 for first GPU, etc.
    model_suffix = ".__unit_test_delete_this_file__.0"
    epoch_model_name = MY_MODEL_NAME + ".epoch_{}" + model_suffix
    success = False
    try:
        model_trainer = train_model_on_data(
            max_epoch, train_iters_per_epoch,
            preproc_sgnn_sklearn_pipeline=None,
            model_trainer=None,
            cuda_device_id=cuda_device_id,
            plot=False,
            epoch_model_name=epoch_model_name
        )
        success = True
    except:
        traceback.print_exc()
    finally:
        epoch_model_name = epoch_model_name.format("{}", "*")
        print(epoch_model_name)
        delete_model(epoch_model_name)
    assert success


def test_can_train_one_epoch_of_one_step_on_GPU():
    batch_size = 10
    train_iters_per_epoch = 1
    max_epoch = 1
    cuda_device_id = 0  # None for CPU, 0 for first GPU, etc.
    model_suffix = ".__unit_test_delete_this_file__.1"
    epoch_model_name = MY_MODEL_NAME + ".epoch_{}" + model_suffix
    success = False
    try:
        model_trainer = train_model_on_data(
            max_epoch, train_iters_per_epoch,
            preproc_sgnn_sklearn_pipeline=None,
            model_trainer=None,
            cuda_device_id=cuda_device_id,
            plot=False,
            epoch_model_name=epoch_model_name
        )
        success = True
    finally:
        epoch_model_name = epoch_model_name.format("{}", "*")
        print(epoch_model_name)
        delete_model(epoch_model_name)  # If this fails, the files were probably not already created.
    assert success


if __name__ == '__main__':
    pytest.main()
