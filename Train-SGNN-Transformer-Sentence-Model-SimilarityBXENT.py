
# coding: utf-8

# # Train an SGNN-Transformer Sentence Model with SimilarityBXENT

# In[5]:


# !pip install joblib
# !echo "joblib" >> requirements.txt
# !pip freeze | grep -i torch >> requirements.txt
# !pip freeze | grep -i numpy >> requirements.txt
get_ipython().system('cat requirements.txt')


# In[2]:


from src.data.read_txt import *
from src.data.config import *
from src.data.training_data import *
from src.data.sgnn_projection_layer import *
from src.model.loss import *
from src.model.transformer import *
from src.model.save_load_model import *
from src.training import *

import numpy as np
from sklearn.metrics import jaccard_similarity_score, f1_score, accuracy_score
from joblib import dump, load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

import math
import copy
import time


# In[ ]:


batch_size = 192
train_iters_per_epoch = 24000
max_epoch = 11
cuda_device_id = 0  # None for CPU, 0 for first GPU, etc.
model_suffix = ".notebook_run.gpu0"
epoch_model_name = MY_MODEL_NAME + ".epoch_{}" + model_suffix
preproc_sgnn_sklearn_pipeline, model_trainer = train_model_on_data(
    max_epoch, train_iters_per_epoch, batch_size,
    preproc_sgnn_sklearn_pipeline=None,
    model_trainer=None,
    cuda_device_id=cuda_device_id,
    plot=False,
    epoch_model_name=epoch_model_name
)


# 
# ## License
# 
# BSD 3-Clause License.
# 
# 
# Copyright (c) 2018, Guillaume Chevalier
# 
# All rights reserved.
# 
