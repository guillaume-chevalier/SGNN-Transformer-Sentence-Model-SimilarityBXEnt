
# coding: utf-8

# # Visualizing/inspecting the learning rate over time and what the model learned

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


# !pip install joblib
# !echo "joblib" >> requirements.txt
# !pip freeze | grep -i torch >> requirements.txt
# !pip freeze | grep -i numpy >> requirements.txt
get_ipython().system('cat requirements.txt')


# In[3]:


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


# In[4]:


batch_size = 160
train_iters_per_epoch = 24000
max_epoch = 11
cuda_device_id = 1  # None for CPU, 0 for first GPU, 1 for second GPU, etc.


# In[5]:


get_ipython().system('ls -1 models_weights/')


# In[6]:


preproc_sgnn_sklearn_pipeline, sentence_projection_model = load_model(
    "my-model{}.epoch_00011.notebook_run.gpu0", cuda_device_id)
# preproc_sgnn_sklearn_pipeline, sentence_projection_model = load_most_recent_model(MY_MODEL_NAME, cuda_device_id)


# In[7]:


model_trainer = TrainerModel(sentence_projection_model)


# ## Visualize the learning rate over time

# In[8]:


# Some code may derive from: https://github.com/harvardnlp/annotated-transformer
# MIT License, Copyright (c) 2018 Alexander Rush

import matplotlib.pyplot as plt
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None),
        get_std_opt(model_trainer)]
plt.plot(
    np.arange(1, train_iters_per_epoch * max_epoch),
    [[opt.rate(i) for opt in opts] for i in range(1, train_iters_per_epoch * max_epoch)]
)
plt.title("Learning rate warmup and decay through the 100k time steps.")
plt.legend(["512:4000", "512:8000", "256:4000", "The one I use"])
plt.show()


# ## Visualize results on some custom data

# In[9]:


sentences_raw = (
    "This is a test. This is another test. "
    "I like bacon. I don't like bacon. "
    "My name is Guillaume. My family name is Chevalier. "
    "Programming can be used for solving complicated math problems. Let's use the Python language to write some scientific code. "
    "My family regrouped for Christmast. We met aunts and uncles. "
    "I like linux. I have an operating system. "
    "Have you ever been in the situation where you've got Jupyter notebooks (iPython notebooks) so huge that you were feeling stuck in your code?. Or even worse: have you ever found yourself duplicating your notebook to do changes, and then ending up with lots of badly named notebooks?. "
    "Either and in any ways. For every medium to big application. "
    "If you're working with notebooks, it is highly likely that you're doing research and development. If doing research and development, to keep your amazing-10x-working-speed-multiplier, it might be a good idea to skip unit tests. "
    "I hope you were satisfied by this reading. What would you do?."
).split(". ")  # each 2 sentence (pairs) above are similar, so we have 10 pairs as such:
category_per_sentence = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

plot_a_result(
    category_per_sentence, cuda_device_id, preproc_sgnn_sklearn_pipeline, 
    sentence_projection_model, sentences_raw)


# The last plot is the expected diagonal block matrix (blocs of 2x2), and the top plot is the prediction. Mid plot is what is above 1 std in the prediction.
