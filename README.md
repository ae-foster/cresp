# On Contrastive Representations of Stochastic Processes [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This project is based on [PyTorch](https://pytorch.org), [Hydra](https://github.com/facebookresearch/hydra) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

## Project organisation
`config/` -> Hydra project configurations

`src/` -> Everything that relates to models

`utils/` -> Helper functions

`experiments/` -> Where experiments are saved (logs, checkpoints, configs, etc)

## Install
```bash
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

# Experiments

## Sinusoids
- Run the different models with:
```bash
python main.py -m +experiment=sine model=ssl,cnp self_attn=id,on  # Untargeted CReSP, FCLR, ANCP, CNP
```
- Figure 2b results can be obtained by varying the *distance between the modes*:
```bash
python main.py -m +experiment=sine model=cnp,ssl self_attn=on dataset.eps=0.0,0.5,1.,2.,5.,8.,10
```
- Figure 2c results can be obtained by varying the *number of training views*:
```bash
python main.py -m +experiment=sine model=cnp,ssl self_attn=on n_views_train=2,5,10,20,50
```

## ShapeNet
- Download and decompress the r2n2 dataset:
```bash
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz -P data
tar zxvf data/ShapeNetRendering.tgz -C data
```
- Run the different models with:
```bash
python main.py +experiment=shapenet model=ssl targeted=True  # Targeted CReSP
python main.py -m +experiment=shapenet model=ssl,cnp self_attn=id,on  # Untargeted CReSP, FCLR, ANCP, CNP
python main.py +experiment=shapenet model=sup fix_clf_train=True  # Supervised
```
- Figure 4a and 4b results can be obtained by varying the *colour distortion strength*:
```bash
python main.py -m +experiment=nocolour self_attn=on,id targeted=False,True model=ssl,cnp
python main.py -m +experiment=shapenet self_attn=on,id targeted=False,True model=ssl,cnp dataset.distortion_s=0.5,1.0,1.5
```
- Figure 4c results can be  obtained by varying the *number of training views*:
```bash
python main.py -m +experiment=shapenet self_attn=on targeted=False self_attn=on,id model=ssl n_views_train=6,12,24
```
- Figure 5a can be  obtained by varying the *fraction of labels available*:
```bash
python main.py -m +experiment=shapenet targeted=False model=ssl,cnp clf.prop=0.01,0.02,0.04,0.1,0.2,0.4,1.0
```
- Figure 5b can be obtained by varying the *number of test views*:
```bash
python main.py -m +experiment=shapenet targeted=False model=ssl,cnp n_views_test=1,2,4,10,20
```

## Snooker
- Table 3 results can be obtained with:
```bash
python main.py +experiment=snooker model=cnp  # CNP
python main.py +experiment=snooker targeted=false  # Untargeted CReSP
python main.py +experiment=snooker  # Targeted CReSP
python main.py +experiment=snooker agg=kernel enc=simple  # MetaCDE
```

# Miscellaneous

## Logging
Can select the logger with `logger=tensorboard`, by default it's using `logger=csv`.

-  Tensorboard: execute the following and forward the port to access live logs:
```bash
tensorboard --logdir experiments/
```

- CSV:
The metrics additionally saved under `.../NAME_OF_EXPERIMENT/RUN_ID/logs/metrics.csv`