
# Train an SGNN-Transformer Sentence Model with SimilarityBXENT


```python
# !pip install joblib
# !echo "joblib" >> requirements.txt
# !pip freeze | grep -i torch >> requirements.txt
# !pip freeze | grep -i numpy >> requirements.txt
!cat requirements.txt
```

    pytest
    pytest-cov
    joblib
    torch==1.0.0
    torchvision==0.2.1
    scikit-learn==0.20.1
    numpy==1.15.4



```python
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
```


```python
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
```

    Epoch 0 Step: 239 Loss: 0.079534 Tokens per Sec: 2505.812324
    Epoch 0 Step: 479 Loss: 0.056101 Tokens per Sec: 2547.705751
    Epoch 0 Step: 719 Loss: 0.067586 Tokens per Sec: 2507.719794
    Epoch 0 Step: 959 Loss: 0.053296 Tokens per Sec: 2546.866083
    Epoch 0 Step: 1199 Loss: 0.060004 Tokens per Sec: 2531.397485
    Epoch 0 Step: 1439 Loss: 0.074048 Tokens per Sec: 2560.327252
    Epoch 0 Step: 1679 Loss: 0.063783 Tokens per Sec: 2538.520846
    Epoch 0 Step: 1919 Loss: 0.079338 Tokens per Sec: 2503.650134
    Epoch 0 Step: 2159 Loss: 0.067302 Tokens per Sec: 2451.060108
    Epoch 0 Step: 2399 Loss: 0.055000 Tokens per Sec: 2438.445192
    Epoch 0 Step: 2639 Loss: 0.073050 Tokens per Sec: 2495.321814
    Epoch 0 Step: 2879 Loss: 0.074002 Tokens per Sec: 2538.956635
    Epoch 0 Step: 3119 Loss: 0.052886 Tokens per Sec: 2535.198840
    Epoch 0 Step: 3359 Loss: 0.094706 Tokens per Sec: 2511.230404
    Epoch 0 Step: 3599 Loss: 0.086621 Tokens per Sec: 2504.209927
    Epoch 0 Step: 3839 Loss: 0.062599 Tokens per Sec: 2549.615593
    Epoch 0 Step: 4079 Loss: 0.057201 Tokens per Sec: 2498.438028
    Epoch 0 Step: 4319 Loss: 0.050393 Tokens per Sec: 2471.556412
    Epoch 0 Step: 4559 Loss: 0.092634 Tokens per Sec: 2317.781912
    Epoch 0 Step: 4799 Loss: 0.050570 Tokens per Sec: 2430.365634
    Epoch 0 Step: 5039 Loss: 0.070621 Tokens per Sec: 2223.213584
    Epoch 0 Step: 5279 Loss: 0.055387 Tokens per Sec: 2267.845056
    Epoch 0 Step: 5519 Loss: 0.054895 Tokens per Sec: 2378.409308
    Epoch 0 Step: 5759 Loss: 0.056352 Tokens per Sec: 2460.902119
    Epoch 0 Step: 5999 Loss: 0.048734 Tokens per Sec: 2505.247648
    Epoch 0 Step: 6239 Loss: 0.049761 Tokens per Sec: 2517.587739
    Epoch 0 Step: 6479 Loss: 0.085300 Tokens per Sec: 2502.584470
    Epoch 0 Step: 6719 Loss: 0.071185 Tokens per Sec: 2431.909109
    Epoch 0 Step: 6959 Loss: 0.055281 Tokens per Sec: 2612.987896
    Epoch 0 Step: 7199 Loss: 0.070359 Tokens per Sec: 2591.764270
    Epoch 0 Step: 7439 Loss: 0.104473 Tokens per Sec: 2483.711086
    Epoch 0 Step: 7679 Loss: 0.061981 Tokens per Sec: 2470.631823
    Epoch 0 Step: 7919 Loss: 0.099785 Tokens per Sec: 2229.011646
    Epoch 0 Step: 8159 Loss: 0.065803 Tokens per Sec: 2239.593568
    Epoch 0 Step: 8399 Loss: 0.061059 Tokens per Sec: 2264.262610
    Epoch 0 Step: 8639 Loss: 0.055098 Tokens per Sec: 2161.498181
    Epoch 0 Step: 8879 Loss: 0.055705 Tokens per Sec: 2228.689178
    Epoch 0 Step: 9119 Loss: 0.082062 Tokens per Sec: 2227.613598
    Epoch 0 Step: 9359 Loss: 0.049592 Tokens per Sec: 2285.374729
    Epoch 0 Step: 9599 Loss: 0.051624 Tokens per Sec: 2249.763762
    Epoch 0 Step: 9839 Loss: 0.074835 Tokens per Sec: 2386.531668
    Epoch 0 Step: 10079 Loss: 0.042173 Tokens per Sec: 2252.571769
    Epoch 0 Step: 10319 Loss: 0.054066 Tokens per Sec: 2467.330407
    Epoch 0 Step: 10559 Loss: 0.052626 Tokens per Sec: 2253.503130
    Epoch 0 Step: 10799 Loss: 0.053746 Tokens per Sec: 2198.063046
    Epoch 0 Step: 11039 Loss: 0.058729 Tokens per Sec: 2385.293927
    Epoch 0 Step: 11279 Loss: 0.058120 Tokens per Sec: 2383.507509
    Epoch 0 Step: 11519 Loss: 0.095185 Tokens per Sec: 2445.510629
    Epoch 0 Step: 11759 Loss: 0.054537 Tokens per Sec: 2589.142023
    Epoch 0 Step: 11999 Loss: 0.050013 Tokens per Sec: 2618.351834
    Epoch 0 Step: 12239 Loss: 0.054317 Tokens per Sec: 2570.767002
    Epoch 0 Step: 12479 Loss: 0.053935 Tokens per Sec: 2619.673368
    Epoch 0 Step: 12719 Loss: 0.048811 Tokens per Sec: 2524.406338
    Epoch 0 Step: 12959 Loss: 0.076213 Tokens per Sec: 2555.651217
    Epoch 0 Step: 13199 Loss: 0.056558 Tokens per Sec: 2546.069112
    Epoch 0 Step: 13439 Loss: 0.060945 Tokens per Sec: 2534.671511
    Epoch 0 Step: 13679 Loss: 0.046313 Tokens per Sec: 2538.236746
    Epoch 0 Step: 13919 Loss: 0.063339 Tokens per Sec: 2524.558100
    Epoch 0 Step: 14159 Loss: 0.058486 Tokens per Sec: 2587.028581
    Epoch 0 Step: 14399 Loss: 0.062366 Tokens per Sec: 2556.519736
    Epoch 0 Step: 14639 Loss: 0.061684 Tokens per Sec: 2544.591846
    Epoch 0 Step: 14879 Loss: 0.054284 Tokens per Sec: 2578.941865
    Epoch 0 Step: 15119 Loss: 0.044014 Tokens per Sec: 2576.370791
    Epoch 0 Step: 15359 Loss: 0.051926 Tokens per Sec: 2573.081321
    Epoch 0 Step: 15599 Loss: 0.050300 Tokens per Sec: 2583.194372
    Epoch 0 Step: 15839 Loss: 0.107517 Tokens per Sec: 2457.503588
    Epoch 0 Step: 16079 Loss: 0.055495 Tokens per Sec: 2464.051710
    Epoch 0 Step: 16319 Loss: 0.059147 Tokens per Sec: 2539.229260
    Epoch 0 Step: 16559 Loss: 0.057288 Tokens per Sec: 2542.852318
    Epoch 0 Step: 16799 Loss: 0.048330 Tokens per Sec: 2495.830751
    Epoch 0 Step: 17039 Loss: 0.055272 Tokens per Sec: 2543.284478
    Epoch 0 Step: 17279 Loss: 0.052810 Tokens per Sec: 2545.078462
    Epoch 0 Step: 17519 Loss: 0.068638 Tokens per Sec: 2562.333719
    Epoch 0 Step: 17759 Loss: 0.069155 Tokens per Sec: 2505.942140
    Epoch 0 Step: 17999 Loss: 0.059448 Tokens per Sec: 2488.280922
    Epoch 0 Step: 18239 Loss: 0.063820 Tokens per Sec: 2559.102607
    Epoch 0 Step: 18479 Loss: 0.048849 Tokens per Sec: 2574.011467
    Epoch 0 Step: 18719 Loss: 0.040472 Tokens per Sec: 2454.670712
    Epoch 0 Step: 18959 Loss: 0.078403 Tokens per Sec: 2311.451801
    Epoch 0 Step: 19199 Loss: 0.046243 Tokens per Sec: 2578.645866
    Epoch 0 Step: 19439 Loss: 0.053910 Tokens per Sec: 2488.903022
    Epoch 0 Step: 19679 Loss: 0.053907 Tokens per Sec: 2591.243415
    Epoch 0 Step: 19919 Loss: 0.050429 Tokens per Sec: 2537.429653
    Epoch 0 Step: 20159 Loss: 0.069737 Tokens per Sec: 2588.422699
    Epoch 0 Step: 20399 Loss: 0.046620 Tokens per Sec: 2481.349192
    Epoch 0 Step: 20639 Loss: 0.057020 Tokens per Sec: 2402.181140
    Epoch 0 Step: 20879 Loss: 0.055819 Tokens per Sec: 2286.475779
    Epoch 0 Step: 21119 Loss: 0.052953 Tokens per Sec: 2447.425468
    Epoch 0 Step: 21359 Loss: 0.070911 Tokens per Sec: 2427.977243
    Epoch 0 Step: 21599 Loss: 0.047939 Tokens per Sec: 2383.048593
    Epoch 0 Step: 21839 Loss: 0.056968 Tokens per Sec: 2512.271914
    Epoch 0 Step: 22079 Loss: 0.057010 Tokens per Sec: 2522.733469
    Epoch 0 Step: 22319 Loss: 0.058249 Tokens per Sec: 2413.868612
    Epoch 0 Step: 22559 Loss: 0.058131 Tokens per Sec: 2422.425778
    Epoch 0 Step: 22799 Loss: 0.053203 Tokens per Sec: 2515.136103
    Epoch 0 Step: 23039 Loss: 0.062443 Tokens per Sec: 2476.631797
    Epoch 0 Step: 23279 Loss: 0.048594 Tokens per Sec: 2361.591185
    Epoch 0 Step: 23519 Loss: 0.059649 Tokens per Sec: 2290.716544
    Epoch 0 Step: 23759 Loss: 0.061884 Tokens per Sec: 2471.002102
    Epoch 0 Step: 23999 Loss: 0.059911 Tokens per Sec: 2523.709854
    2019-01-07 10:05:32  - Saved model to files: ./models_weights/my-model.sklearn.epoch_00000.notebook_run.gpu0 ./models_weights/my-model.pytorch.epoch_00000.notebook_run.gpu0
    Epoch 1 Step: 239 Loss: 0.051407 Tokens per Sec: 2390.558596
    Epoch 1 Step: 479 Loss: 0.045738 Tokens per Sec: 2362.694753
    Epoch 1 Step: 719 Loss: 0.051654 Tokens per Sec: 2486.625190
    Epoch 1 Step: 959 Loss: 0.040367 Tokens per Sec: 2524.012741
    Epoch 1 Step: 1199 Loss: 0.040316 Tokens per Sec: 2430.297052
    Epoch 1 Step: 1439 Loss: 0.048170 Tokens per Sec: 2533.374415
    Epoch 1 Step: 1679 Loss: 0.060015 Tokens per Sec: 2478.656436
    Epoch 1 Step: 1919 Loss: 0.062205 Tokens per Sec: 2539.822601
    Epoch 1 Step: 2159 Loss: 0.049340 Tokens per Sec: 2440.444101
    Epoch 1 Step: 2399 Loss: 0.053292 Tokens per Sec: 2336.529812
    Epoch 1 Step: 2639 Loss: 0.053643 Tokens per Sec: 2457.081784
    Epoch 1 Step: 2879 Loss: 0.049491 Tokens per Sec: 2467.149794
    Epoch 1 Step: 3119 Loss: 0.049554 Tokens per Sec: 2513.474265
    Epoch 1 Step: 3359 Loss: 0.055617 Tokens per Sec: 2498.678445
    Epoch 1 Step: 3599 Loss: 0.068749 Tokens per Sec: 2530.501461
    Epoch 1 Step: 3839 Loss: 0.062038 Tokens per Sec: 2488.057801
    Epoch 1 Step: 4079 Loss: 0.051340 Tokens per Sec: 2496.455215
    Epoch 1 Step: 4319 Loss: 0.060002 Tokens per Sec: 2465.832294
    Epoch 1 Step: 4559 Loss: 0.059636 Tokens per Sec: 2536.194363
    Epoch 1 Step: 4799 Loss: 0.055613 Tokens per Sec: 2523.169417
    Epoch 1 Step: 5039 Loss: 0.070489 Tokens per Sec: 2479.858640
    Epoch 1 Step: 5279 Loss: 0.065318 Tokens per Sec: 2521.110593
    Epoch 1 Step: 5519 Loss: 0.044503 Tokens per Sec: 2532.898824
    Epoch 1 Step: 5759 Loss: 0.061802 Tokens per Sec: 2503.015569
    Epoch 1 Step: 5999 Loss: 0.046447 Tokens per Sec: 2559.286358
    Epoch 1 Step: 6239 Loss: 0.050959 Tokens per Sec: 2540.295324
    Epoch 1 Step: 6479 Loss: 0.044718 Tokens per Sec: 2560.756584
    Epoch 1 Step: 6719 Loss: 0.083031 Tokens per Sec: 2475.607919
    Epoch 1 Step: 6959 Loss: 0.050817 Tokens per Sec: 2458.621345
    Epoch 1 Step: 7199 Loss: 0.058283 Tokens per Sec: 2483.761672
    Epoch 1 Step: 7439 Loss: 0.041650 Tokens per Sec: 2565.915078
    Epoch 1 Step: 7679 Loss: 0.054724 Tokens per Sec: 2570.761111
    Epoch 1 Step: 7919 Loss: 0.063471 Tokens per Sec: 2581.614505
    Epoch 1 Step: 8159 Loss: 0.041398 Tokens per Sec: 2505.112011
    Epoch 1 Step: 8399 Loss: 0.067593 Tokens per Sec: 2515.197903
    Epoch 1 Step: 8639 Loss: 0.069686 Tokens per Sec: 2382.403511
    Epoch 1 Step: 8879 Loss: 0.052854 Tokens per Sec: 2432.915548
    Epoch 1 Step: 9119 Loss: 0.063079 Tokens per Sec: 2556.271612
    Epoch 1 Step: 9359 Loss: 0.053092 Tokens per Sec: 2636.683076
    Epoch 1 Step: 9599 Loss: 0.051026 Tokens per Sec: 2478.364087
    Epoch 1 Step: 9839 Loss: 0.046112 Tokens per Sec: 2519.274698
    Epoch 1 Step: 10079 Loss: 0.067315 Tokens per Sec: 2521.368723
    Epoch 1 Step: 10319 Loss: 0.065241 Tokens per Sec: 2527.254976
    Epoch 1 Step: 10559 Loss: 0.050516 Tokens per Sec: 2547.052082
    Epoch 1 Step: 10799 Loss: 0.078200 Tokens per Sec: 2499.352493
    Epoch 1 Step: 11039 Loss: 0.047257 Tokens per Sec: 2551.077381
    Epoch 1 Step: 11279 Loss: 0.049722 Tokens per Sec: 2551.308482
    Epoch 1 Step: 11519 Loss: 0.068343 Tokens per Sec: 2571.711724
    Epoch 1 Step: 11759 Loss: 0.047196 Tokens per Sec: 2559.700910
    Epoch 1 Step: 11999 Loss: 0.066788 Tokens per Sec: 2577.741516
    Epoch 1 Step: 12239 Loss: 0.054743 Tokens per Sec: 2549.516416
    Epoch 1 Step: 12479 Loss: 0.068151 Tokens per Sec: 2479.855004
    Epoch 1 Step: 12719 Loss: 0.047784 Tokens per Sec: 2545.989345
    Epoch 1 Step: 12959 Loss: 0.095695 Tokens per Sec: 2534.498852
    Epoch 1 Step: 13199 Loss: 0.069505 Tokens per Sec: 2633.619189
    Epoch 1 Step: 13439 Loss: 0.083179 Tokens per Sec: 2560.543702
    Epoch 1 Step: 13679 Loss: 0.066503 Tokens per Sec: 2588.918321
    Epoch 1 Step: 13919 Loss: 0.080156 Tokens per Sec: 2529.527799
    Epoch 1 Step: 14159 Loss: 0.063771 Tokens per Sec: 2546.836344
    Epoch 1 Step: 14399 Loss: 0.060351 Tokens per Sec: 2593.496200
    Epoch 1 Step: 14639 Loss: 0.067953 Tokens per Sec: 2569.206589
    Epoch 1 Step: 14879 Loss: 0.113040 Tokens per Sec: 2572.560768
    Epoch 1 Step: 15119 Loss: 0.046268 Tokens per Sec: 2595.724925
    Epoch 1 Step: 15359 Loss: 0.062603 Tokens per Sec: 2531.021319
    Epoch 1 Step: 15599 Loss: 0.065947 Tokens per Sec: 2553.534655
    Epoch 1 Step: 15839 Loss: 0.079001 Tokens per Sec: 2597.479194
    Epoch 1 Step: 16079 Loss: 0.063620 Tokens per Sec: 2546.714638
    Epoch 1 Step: 16319 Loss: 0.058196 Tokens per Sec: 2535.190095
    Epoch 1 Step: 16559 Loss: 0.059678 Tokens per Sec: 2536.678588
    Epoch 1 Step: 16799 Loss: 0.058145 Tokens per Sec: 2550.476096
    Epoch 1 Step: 17039 Loss: 0.060396 Tokens per Sec: 2575.319038
    Epoch 1 Step: 17279 Loss: 0.080552 Tokens per Sec: 2572.471047
    Epoch 1 Step: 17519 Loss: 0.071285 Tokens per Sec: 2529.707258
    Epoch 1 Step: 17759 Loss: 0.056895 Tokens per Sec: 2547.426151
    Epoch 1 Step: 17999 Loss: 0.048218 Tokens per Sec: 2593.063103
    Epoch 1 Step: 18239 Loss: 0.056778 Tokens per Sec: 2537.065968
    Epoch 1 Step: 18479 Loss: 0.058151 Tokens per Sec: 2529.731622
    Epoch 1 Step: 18719 Loss: 0.054339 Tokens per Sec: 2618.866030
    Epoch 1 Step: 18959 Loss: 0.063607 Tokens per Sec: 2648.893333
    Epoch 1 Step: 19199 Loss: 0.066775 Tokens per Sec: 2547.849351
    Epoch 1 Step: 19439 Loss: 0.082401 Tokens per Sec: 2603.857850
    Epoch 1 Step: 19679 Loss: 0.074732 Tokens per Sec: 2502.425841
    Epoch 1 Step: 19919 Loss: 0.059640 Tokens per Sec: 2534.606155
    Epoch 1 Step: 20159 Loss: 0.060950 Tokens per Sec: 2458.764701
    Epoch 1 Step: 20399 Loss: 0.052764 Tokens per Sec: 2526.204512
    Epoch 1 Step: 20639 Loss: 0.044887 Tokens per Sec: 2521.885355
    Epoch 1 Step: 20879 Loss: 0.061172 Tokens per Sec: 2444.303380
    Epoch 1 Step: 21119 Loss: 0.067961 Tokens per Sec: 2468.536441
    Epoch 1 Step: 21359 Loss: 0.060509 Tokens per Sec: 2453.680797
    Epoch 1 Step: 21599 Loss: 0.055478 Tokens per Sec: 2435.997056
    Epoch 1 Step: 21839 Loss: 0.058443 Tokens per Sec: 2327.632899
    Epoch 1 Step: 22079 Loss: 0.056243 Tokens per Sec: 2377.210601
    Epoch 1 Step: 22319 Loss: 0.066709 Tokens per Sec: 2331.465677
    Epoch 1 Step: 22559 Loss: 0.061322 Tokens per Sec: 2415.237599
    Epoch 1 Step: 22799 Loss: 0.056875 Tokens per Sec: 2346.795214
    Epoch 1 Step: 23039 Loss: 0.065339 Tokens per Sec: 2439.464302
    Epoch 1 Step: 23279 Loss: 0.053481 Tokens per Sec: 2396.067003
    Epoch 1 Step: 23519 Loss: 0.050586 Tokens per Sec: 2418.514226
    Epoch 1 Step: 23759 Loss: 0.045736 Tokens per Sec: 2407.379213
    Epoch 1 Step: 23999 Loss: 0.042582 Tokens per Sec: 2421.742509
    2019-01-07 15:02:49  - Saved model to files: ./models_weights/my-model.sklearn.epoch_00001.notebook_run.gpu0 ./models_weights/my-model.pytorch.epoch_00001.notebook_run.gpu0
    Epoch 2 Step: 239 Loss: 0.046688 Tokens per Sec: 2334.749321
    Epoch 2 Step: 479 Loss: 0.060771 Tokens per Sec: 2338.566829
    Epoch 2 Step: 719 Loss: 0.084039 Tokens per Sec: 2437.313869
    Epoch 2 Step: 959 Loss: 0.062754 Tokens per Sec: 2369.955917
    Epoch 2 Step: 1199 Loss: 0.067327 Tokens per Sec: 2405.558306
    Epoch 2 Step: 1439 Loss: 0.050225 Tokens per Sec: 2500.664551
    Epoch 2 Step: 1679 Loss: 0.073147 Tokens per Sec: 2432.848753
    Epoch 2 Step: 1919 Loss: 0.050865 Tokens per Sec: 2420.266216
    Epoch 2 Step: 2159 Loss: 0.059185 Tokens per Sec: 2370.081525
    Epoch 2 Step: 2399 Loss: 0.049397 Tokens per Sec: 2404.324433
    Epoch 2 Step: 2639 Loss: 0.060262 Tokens per Sec: 2359.627281
    Epoch 2 Step: 2879 Loss: 0.046172 Tokens per Sec: 2425.853783
    Epoch 2 Step: 3119 Loss: 0.055477 Tokens per Sec: 2423.556707
    Epoch 2 Step: 3359 Loss: 0.051570 Tokens per Sec: 2487.320221
    Epoch 2 Step: 3599 Loss: 0.051201 Tokens per Sec: 2375.685673
    Epoch 2 Step: 3839 Loss: 0.061646 Tokens per Sec: 2473.969128
    Epoch 2 Step: 4079 Loss: 0.050882 Tokens per Sec: 2465.188843
    Epoch 2 Step: 4319 Loss: 0.064120 Tokens per Sec: 2427.187288
    Epoch 2 Step: 4559 Loss: 0.074446 Tokens per Sec: 2417.360452
    Epoch 2 Step: 4799 Loss: 0.044102 Tokens per Sec: 2461.714460
    Epoch 2 Step: 5039 Loss: 0.052772 Tokens per Sec: 2425.832669
    Epoch 2 Step: 5279 Loss: 0.091840 Tokens per Sec: 2446.275124
    Epoch 2 Step: 5519 Loss: 0.053395 Tokens per Sec: 2528.682993
    Epoch 2 Step: 5759 Loss: 0.093565 Tokens per Sec: 2482.002705
    Epoch 2 Step: 5999 Loss: 0.055923 Tokens per Sec: 2366.503512
    Epoch 2 Step: 6239 Loss: 0.072142 Tokens per Sec: 2491.821111
    Epoch 2 Step: 6479 Loss: 0.057194 Tokens per Sec: 2470.325670
    Epoch 2 Step: 6719 Loss: 0.042567 Tokens per Sec: 2401.895461
    Epoch 2 Step: 6959 Loss: 0.057160 Tokens per Sec: 2377.720727
    Epoch 2 Step: 7199 Loss: 0.055164 Tokens per Sec: 2486.031797
    Epoch 2 Step: 7439 Loss: 0.042246 Tokens per Sec: 2526.159291
    Epoch 2 Step: 7679 Loss: 0.065562 Tokens per Sec: 2500.771012
    Epoch 2 Step: 7919 Loss: 0.067350 Tokens per Sec: 2447.333133
    Epoch 2 Step: 8159 Loss: 0.091257 Tokens per Sec: 2455.853030
    Epoch 2 Step: 8399 Loss: 0.055721 Tokens per Sec: 2435.480819
    Epoch 2 Step: 8639 Loss: 0.056641 Tokens per Sec: 2432.272113
    Epoch 2 Step: 8879 Loss: 0.075930 Tokens per Sec: 2458.170303
    Epoch 2 Step: 9119 Loss: 0.050899 Tokens per Sec: 2462.053520
    Epoch 2 Step: 9359 Loss: 0.047876 Tokens per Sec: 2458.868336
    Epoch 2 Step: 9599 Loss: 0.054494 Tokens per Sec: 2418.630576
    Epoch 2 Step: 9839 Loss: 0.067040 Tokens per Sec: 2352.908012
    Epoch 2 Step: 10079 Loss: 0.060733 Tokens per Sec: 2506.273102
    Epoch 2 Step: 10319 Loss: 0.050877 Tokens per Sec: 2510.717972
    Epoch 2 Step: 10559 Loss: 0.063271 Tokens per Sec: 2463.343620
    Epoch 2 Step: 10799 Loss: 0.044971 Tokens per Sec: 2471.873992
    Epoch 2 Step: 11039 Loss: 0.049862 Tokens per Sec: 2452.585538
    Epoch 2 Step: 11279 Loss: 0.062935 Tokens per Sec: 2399.507766
    Epoch 2 Step: 11519 Loss: 0.056260 Tokens per Sec: 2468.628162
    Epoch 2 Step: 11759 Loss: 0.052879 Tokens per Sec: 2398.604098
    Epoch 2 Step: 11999 Loss: 0.044189 Tokens per Sec: 2428.113216
    Epoch 2 Step: 12239 Loss: 0.082285 Tokens per Sec: 2383.342534
    Epoch 2 Step: 12479 Loss: 0.056648 Tokens per Sec: 2466.335210
    Epoch 2 Step: 12719 Loss: 0.050378 Tokens per Sec: 2427.795649
    Epoch 2 Step: 12959 Loss: 0.050373 Tokens per Sec: 2446.028509
    Epoch 2 Step: 13199 Loss: 0.084315 Tokens per Sec: 2421.038602
    Epoch 2 Step: 13439 Loss: 0.048842 Tokens per Sec: 2491.418401
    Epoch 2 Step: 13679 Loss: 0.052029 Tokens per Sec: 2464.202697
    Epoch 2 Step: 13919 Loss: 0.065698 Tokens per Sec: 2591.270501
    Epoch 2 Step: 14159 Loss: 0.050559 Tokens per Sec: 2484.746421
    Epoch 2 Step: 14399 Loss: 0.052026 Tokens per Sec: 2415.585749
    Epoch 2 Step: 14639 Loss: 0.047790 Tokens per Sec: 2351.825495
    Epoch 2 Step: 14879 Loss: 0.052695 Tokens per Sec: 2520.705337
    Epoch 2 Step: 15119 Loss: 0.048949 Tokens per Sec: 2410.753010
    Epoch 2 Step: 15359 Loss: 0.049319 Tokens per Sec: 2444.003767
    Epoch 2 Step: 15599 Loss: 0.051183 Tokens per Sec: 2481.253671
    Epoch 2 Step: 15839 Loss: 0.049587 Tokens per Sec: 2459.634469
    Epoch 2 Step: 16079 Loss: 0.068763 Tokens per Sec: 2447.336916
    Epoch 2 Step: 16319 Loss: 0.050009 Tokens per Sec: 2486.289250
    Epoch 2 Step: 16559 Loss: 0.054445 Tokens per Sec: 2356.981375
    Epoch 2 Step: 16799 Loss: 0.059054 Tokens per Sec: 2341.790913
    Epoch 2 Step: 17039 Loss: 0.066519 Tokens per Sec: 2438.512803
    Epoch 2 Step: 17279 Loss: 0.051239 Tokens per Sec: 2461.575375
    Epoch 2 Step: 17519 Loss: 0.055436 Tokens per Sec: 2459.725543
    Epoch 2 Step: 17759 Loss: 0.056573 Tokens per Sec: 2379.570515
    Epoch 2 Step: 17999 Loss: 0.056936 Tokens per Sec: 2403.272830
    Epoch 2 Step: 18239 Loss: 0.071790 Tokens per Sec: 2428.972721
    Epoch 2 Step: 18479 Loss: 0.058107 Tokens per Sec: 2316.866480
    Epoch 2 Step: 18719 Loss: 0.055148 Tokens per Sec: 2393.143962
    Epoch 2 Step: 18959 Loss: 0.062460 Tokens per Sec: 2424.998935
    Epoch 2 Step: 19199 Loss: 0.056545 Tokens per Sec: 2405.050859
    Epoch 2 Step: 19439 Loss: 0.075046 Tokens per Sec: 2408.463331
    Epoch 2 Step: 19679 Loss: 0.044282 Tokens per Sec: 2418.294812
    Epoch 2 Step: 19919 Loss: 0.058876 Tokens per Sec: 2325.659865
    Epoch 2 Step: 20159 Loss: 0.063109 Tokens per Sec: 2336.629202
    Epoch 2 Step: 20399 Loss: 0.051702 Tokens per Sec: 2339.954408
    Epoch 2 Step: 20639 Loss: 0.054136 Tokens per Sec: 2385.340782
    Epoch 2 Step: 20879 Loss: 0.051736 Tokens per Sec: 2351.338167
    Epoch 2 Step: 21119 Loss: 0.067850 Tokens per Sec: 2361.307043
    Epoch 2 Step: 21359 Loss: 0.054586 Tokens per Sec: 2307.885518
    Epoch 2 Step: 21599 Loss: 0.057267 Tokens per Sec: 2409.109919
    Epoch 2 Step: 21839 Loss: 0.071078 Tokens per Sec: 2351.896344
    Epoch 2 Step: 22079 Loss: 0.058159 Tokens per Sec: 2326.434952
    Epoch 2 Step: 22319 Loss: 0.058208 Tokens per Sec: 2339.470203
    Epoch 2 Step: 22559 Loss: 0.053033 Tokens per Sec: 2409.223971
    Epoch 2 Step: 22799 Loss: 0.054263 Tokens per Sec: 2269.134667
    Epoch 2 Step: 23039 Loss: 0.062451 Tokens per Sec: 2453.574068
    Epoch 2 Step: 23279 Loss: 0.055998 Tokens per Sec: 2397.037174
    Epoch 2 Step: 23519 Loss: 0.051290 Tokens per Sec: 2351.647618
    Epoch 2 Step: 23759 Loss: 0.053839 Tokens per Sec: 2448.545267
    Epoch 2 Step: 23999 Loss: 0.065624 Tokens per Sec: 2426.751436
    2019-01-07 20:12:47  - Saved model to files: ./models_weights/my-model.sklearn.epoch_00002.notebook_run.gpu0 ./models_weights/my-model.pytorch.epoch_00002.notebook_run.gpu0
    Epoch 3 Step: 239 Loss: 0.050503 Tokens per Sec: 2206.911807
    Epoch 3 Step: 479 Loss: 0.056120 Tokens per Sec: 2231.597249
    Epoch 3 Step: 719 Loss: 0.066362 Tokens per Sec: 2265.060222
    Epoch 3 Step: 959 Loss: 0.075225 Tokens per Sec: 2252.288086
    Epoch 3 Step: 1199 Loss: 0.046970 Tokens per Sec: 2322.741021
    Epoch 3 Step: 1439 Loss: 0.059510 Tokens per Sec: 2284.620380
    Epoch 3 Step: 1679 Loss: 0.059562 Tokens per Sec: 2338.936927
    Epoch 3 Step: 1919 Loss: 0.054074 Tokens per Sec: 2278.632688
    Epoch 3 Step: 2159 Loss: 0.070618 Tokens per Sec: 2299.928325
    Epoch 3 Step: 2399 Loss: 0.053511 Tokens per Sec: 2322.374124
    Epoch 3 Step: 2639 Loss: 0.042013 Tokens per Sec: 2424.040278
    Epoch 3 Step: 2879 Loss: 0.053936 Tokens per Sec: 2311.101310
    Epoch 3 Step: 3119 Loss: 0.069536 Tokens per Sec: 2346.512658
    Epoch 3 Step: 3359 Loss: 0.046956 Tokens per Sec: 2388.928893
    Epoch 3 Step: 3599 Loss: 0.051383 Tokens per Sec: 2372.830809
    Epoch 3 Step: 3839 Loss: 0.043823 Tokens per Sec: 2377.678152
    Epoch 3 Step: 4079 Loss: 0.064916 Tokens per Sec: 2357.338115
    Epoch 3 Step: 4319 Loss: 0.044692 Tokens per Sec: 2417.295109
    Epoch 3 Step: 4559 Loss: 0.055474 Tokens per Sec: 2274.987232
    Epoch 3 Step: 4799 Loss: 0.061781 Tokens per Sec: 2323.988887
    Epoch 3 Step: 5039 Loss: 0.053243 Tokens per Sec: 2326.915328
    Epoch 3 Step: 5279 Loss: 0.056134 Tokens per Sec: 2314.329776
    Epoch 3 Step: 5519 Loss: 0.053381 Tokens per Sec: 2324.536284
    Epoch 3 Step: 5759 Loss: 0.059614 Tokens per Sec: 2388.278434
    Epoch 3 Step: 5999 Loss: 0.056721 Tokens per Sec: 2303.347859
    Epoch 3 Step: 6239 Loss: 0.074365 Tokens per Sec: 2378.476925
    Epoch 3 Step: 6479 Loss: 0.035591 Tokens per Sec: 2375.891351
    Epoch 3 Step: 6719 Loss: 0.071318 Tokens per Sec: 2318.768479
    [...] up to epoch 11


## License

BSD 3-Clause License.


Copyright (c) 2018, Guillaume Chevalier

All rights reserved.


# Visualizing/inspecting the learning rate over time and what the model learned


```python
!nvidia-smi
```

    Mon Jan 14 03:57:40 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
    |  0%   32C    P8    17W / 280W |   9470MiB / 11178MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  GeForce GTX 108...  Off  | 00000000:81:00.0 Off |                  N/A |
    |  0%   30C    P8    16W / 280W |     10MiB / 11178MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      9591      C   ...o/miniconda3/envs/tensorflow/bin/python   737MiB |
    |    0     14274      C   ...o/miniconda3/envs/tensorflow/bin/python  8723MiB |
    +-----------------------------------------------------------------------------+



```python
# !pip install joblib
# !echo "joblib" >> requirements.txt
# !pip freeze | grep -i torch >> requirements.txt
# !pip freeze | grep -i numpy >> requirements.txt
!cat requirements.txt
```

    pytest
    pytest-cov
    joblib
    torch==1.0.0
    torchvision==0.2.1
    scikit-learn==0.20.1
    numpy==1.15.4



```python
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
```


```python
batch_size = 160
train_iters_per_epoch = 24000
max_epoch = 11
cuda_device_id = 1  # None for CPU, 0 for first GPU, 1 for second GPU, etc.
```


```python
!ls -1 models_weights/
```

    my-model.pytorch.epoch_00000.notebook_run.gpu0
    my-model.pytorch.epoch_00000.notebook_run.gpu1
    my-model.pytorch.epoch_00001.notebook_run.gpu0
    my-model.pytorch.epoch_00001.notebook_run.gpu1
    my-model.pytorch.epoch_00002.notebook_run.gpu0
    my-model.pytorch.epoch_00002.notebook_run.gpu1
    my-model.pytorch.epoch_00003.notebook_run.gpu0
    my-model.pytorch.epoch_00003.notebook_run.gpu1
    my-model.pytorch.epoch_00004.notebook_run.gpu0
    my-model.pytorch.epoch_00004.notebook_run.gpu1
    my-model.pytorch.epoch_00005.notebook_run.gpu0
    my-model.pytorch.epoch_00005.notebook_run.gpu1
    my-model.pytorch.epoch_00006.notebook_run.gpu0
    my-model.pytorch.epoch_00006.notebook_run.gpu1
    my-model.pytorch.epoch_00007.notebook_run.gpu0
    my-model.pytorch.epoch_00007.notebook_run.gpu1
    my-model.pytorch.epoch_00008.notebook_run.gpu0
    my-model.pytorch.epoch_00008.notebook_run.gpu1
    my-model.pytorch.epoch_00009.notebook_run.gpu0
    my-model.pytorch.epoch_00009.notebook_run.gpu1
    my-model.pytorch.epoch_00010.notebook_run.gpu0
    my-model.pytorch.epoch_00010.notebook_run.gpu1
    my-model.pytorch.epoch_00011.notebook_run.gpu0
    my-model.pytorch.epoch_00011.notebook_run.gpu1
    my-model.pytorch.epoch_00012.notebook_run.gpu1
    my-model.pytorch.epoch_00013.notebook_run.gpu1
    my-model.pytorch.epoch_00014.notebook_run.gpu1
    my-model.pytorch.epoch_00015.notebook_run.gpu1
    my-model.pytorch.epoch_00016.notebook_run.gpu1
    my-model.pytorch.epoch_00017.notebook_run.gpu1
    my-model.pytorch.epoch_00018.notebook_run.gpu1
    my-model.pytorch.epoch_00019.notebook_run.gpu1
    my-model.pytorch.epoch_00020.notebook_run.gpu1
    my-model.pytorch.epoch_00021.notebook_run.gpu1
    my-model.pytorch.epoch_00022.notebook_run.gpu1
    my-model.pytorch.epoch_00023.notebook_run.gpu1
    my-model.sklearn.epoch_00000.notebook_run.gpu0
    my-model.sklearn.epoch_00000.notebook_run.gpu1
    my-model.sklearn.epoch_00001.notebook_run.gpu0
    my-model.sklearn.epoch_00001.notebook_run.gpu1
    my-model.sklearn.epoch_00002.notebook_run.gpu0
    my-model.sklearn.epoch_00002.notebook_run.gpu1
    my-model.sklearn.epoch_00003.notebook_run.gpu0
    my-model.sklearn.epoch_00003.notebook_run.gpu1
    my-model.sklearn.epoch_00004.notebook_run.gpu0
    my-model.sklearn.epoch_00004.notebook_run.gpu1
    my-model.sklearn.epoch_00005.notebook_run.gpu0
    my-model.sklearn.epoch_00005.notebook_run.gpu1
    my-model.sklearn.epoch_00006.notebook_run.gpu0
    my-model.sklearn.epoch_00006.notebook_run.gpu1
    my-model.sklearn.epoch_00007.notebook_run.gpu0
    my-model.sklearn.epoch_00007.notebook_run.gpu1
    my-model.sklearn.epoch_00008.notebook_run.gpu0
    my-model.sklearn.epoch_00008.notebook_run.gpu1
    my-model.sklearn.epoch_00009.notebook_run.gpu0
    my-model.sklearn.epoch_00009.notebook_run.gpu1
    my-model.sklearn.epoch_00010.notebook_run.gpu0
    my-model.sklearn.epoch_00010.notebook_run.gpu1
    my-model.sklearn.epoch_00011.notebook_run.gpu0
    my-model.sklearn.epoch_00011.notebook_run.gpu1
    my-model.sklearn.epoch_00012.notebook_run.gpu1
    my-model.sklearn.epoch_00013.notebook_run.gpu1
    my-model.sklearn.epoch_00014.notebook_run.gpu1
    my-model.sklearn.epoch_00015.notebook_run.gpu1
    my-model.sklearn.epoch_00016.notebook_run.gpu1
    my-model.sklearn.epoch_00017.notebook_run.gpu1
    my-model.sklearn.epoch_00018.notebook_run.gpu1
    my-model.sklearn.epoch_00019.notebook_run.gpu1
    my-model.sklearn.epoch_00020.notebook_run.gpu1
    my-model.sklearn.epoch_00021.notebook_run.gpu1
    my-model.sklearn.epoch_00022.notebook_run.gpu1
    my-model.sklearn.epoch_00023.notebook_run.gpu1



```python
preproc_sgnn_sklearn_pipeline, sentence_projection_model = load_model(
    "my-model{}.epoch_00011.notebook_run.gpu0", cuda_device_id)
# preproc_sgnn_sklearn_pipeline, sentence_projection_model = load_most_recent_model(MY_MODEL_NAME, cuda_device_id)
```

    Loaded model from files: ./models_weights/my-model.sklearn.epoch_00011.notebook_run.gpu0 ./models_weights/my-model.pytorch.epoch_00011.notebook_run.gpu0



```python
model_trainer = TrainerModel(sentence_projection_model)
```

## Visualize the learning rate over time


```python
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
```


![png](Load-and-Inspect-Model-Predictions_files/Load-and-Inspect-Model-Predictions_9_0.png)


## Visualize results on some custom data


```python
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
```

    Matrices are symmetric, so on each border is a sentence dotted with annother one in similarity to get something that is almost like covariance of each sentences to each other. We should observe 2x2 activated blocks along the diagonal. The loss function is a binary cross-entropy on this sentence-to-sentence similarity grid we see. I seem to have invented a new similarity loss function but it probably already exists...



![png](Load-and-Inspect-Model-Predictions_files/Load-and-Inspect-Model-Predictions_11_1.png)



![png](Load-and-Inspect-Model-Predictions_files/Load-and-Inspect-Model-Predictions_11_2.png)



![png](Load-and-Inspect-Model-Predictions_files/Load-and-Inspect-Model-Predictions_11_3.png)


    Compute the 2D overlap in the matrix:
    test_jaccard_score: 0.885
    test_f1_score: 0.5306122448979592
    test_accuracy_score: 0.885


The last plot is the expected diagonal block matrix (blocs of 2x2), and the top plot is the prediction. Mid plot is what is above 1 std in the prediction.
