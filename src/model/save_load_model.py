import datetime
import os
import glob

from joblib import dump, load
import torch

MY_MODEL_NAME = "my-model{}"
DEFAULT_FOLDER = "./models_weights"


def save_model(preproc_sgnn_sklearn_pipeline, sentence_projection_model, model_name=MY_MODEL_NAME):
    a = os.path.join(DEFAULT_FOLDER, model_name.format(".sklearn"))
    b = os.path.join(DEFAULT_FOLDER, model_name.format(".pytorch"))
    if not os.path.exists(DEFAULT_FOLDER):
        os.mkdir(DEFAULT_FOLDER)
    dump(preproc_sgnn_sklearn_pipeline, a)
    with open(b, "wb") as f:
        torch.save(sentence_projection_model, f=f)
    print(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        " - Saved model to files:", a, b
    )


def load_model(model_name=MY_MODEL_NAME, cuda_device_id=None):
    a = os.path.join(DEFAULT_FOLDER, model_name.format(".sklearn"))
    b = os.path.join(DEFAULT_FOLDER, model_name.format(".pytorch"))
    preproc_sgnn_sklearn_pipeline = load(a)
    sentence_projection_model = torch.load(b)
    if cuda_device_id is not None:
        sentence_projection_model = sentence_projection_model.cuda(cuda_device_id)
    print("Loaded model from files:", a, b)
    return preproc_sgnn_sklearn_pipeline, sentence_projection_model


def delete_model(epoch_model_name):
    print("Deleting model: {}".format(epoch_model_name))
    epoch_model_name = epoch_model_name.format("*")
    path = os.path.join(DEFAULT_FOLDER, epoch_model_name)
    files = glob.glob(path)
    assert len(files) == 2, (
        "Test error: you may want to check these files out to delete them yourself: {}. The path was: {}".format(
            files, path))
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def load_most_recent_model(model_name, cuda_device_id):
    a = model_name.format(".sklearn*")
    b = model_name.format(".pytorch*")
    a = list(sorted(glob.glob(os.path.join(DEFAULT_FOLDER, a))))[-1]  # model with highest epoch number
    b = list(sorted(glob.glob(os.path.join(DEFAULT_FOLDER, b))))[-1]  # model with highest epoch number

    suffix = a.split(model_name.format(".sklearn"))[-1]
    return load_model(model_name + suffix, cuda_device_id)
