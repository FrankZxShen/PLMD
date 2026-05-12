# PLMD: Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation

[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b7bec.svg)](#citation)
[![Task](https://img.shields.io/badge/Task-Universal%20Goal--Oriented%20Navigation-20bf6b.svg)](#overview)
[![Method](https://img.shields.io/badge/Method-Label%20Map%20Diffusion-f7b731.svg)](#overview)

Official code release for the ICML 2026 paper:

**Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation**

PLMD is a plug-and-play diffusion framework over label maps for universal goal-oriented navigation. This repository contains the code paths used for map collection, diffusion-model training, and evaluation across ObjectNav (ON), Instance ImageNav (IIN), and Multi-Robot ObjectNav (MRON).

## Overview

PLMD separates goal-oriented navigation into reusable components:

- `PLMD-MapCollection`: label-map collection pipeline.
- `PLMD-training`: obstacle-map and semantic-map diffusion training.
- `PLMD-ON`: ObjectNav evaluation.
- `PLMD-IIN`: Instance ImageNav evaluation.
- `PLMD-MRON`: Multi-Robot ObjectNav evaluation.
- `Navigation_Videos`: qualitative navigation videos and process visualizations.

## Dependencies

- OS: Ubuntu 20.04.6
- CUDA: 12.1
- cuDNN: 8.5.0
- Python 3
- PyTorch >= 1.13.0
- Python packages:

```bash
pip install -r requirements.txt
```

Each task directory also contains task-specific setup notes.

## Label Map Collection

Follow `PLMD-MapCollection/README.md` to install the environment.

Run:

```bash
cd PLMD-MapCollection
python main_MapCollection.py --visualize 2 \
    --not_explore 0 \
    --task_config tasks/objectnav_hm3d_1agent.yaml
```

## PLMD Training

### Obstacle Map Model

Replace `PLMD-training/options/train/ir-sde.yml` line 15 and line 22 with your obstacle-map mask dataset path and obstacle-map dataset path.

Run:

```bash
cd PLMD-training
python train.py
```

### Semantic Map Model

Replace `PLMD-training/options/train/ir-sde.yml` line 15 and line 22 with your semantic-map mask dataset path and semantic-map dataset path.

Run:

```bash
cd PLMD-training
python train.py
```

Pre-trained weights will be provided with the official release.

## ON Eval

Follow `PLMD-ON/README.md` to install the ON environment.

Replace `PLMD-ON/configs/local.yml` lines 44 and 45 with the obstacle-map model path and semantic-map model path.

Run:

```bash
cd PLMD-ON
python main_ON.py -d ./ON_EXP/ \
    --visualize 2 \
    --num_sem_categories 16 \
    --not_explore 1 \
    --task_config tasks/objectnav_hm3d_1agent.yaml
```

## IIN Eval

Follow `PLMD-IIN/README.md` to install the IIN environment.

Replace `PLMD-IIN/configs/local.yml` lines 44 and 45 with the obstacle-map model path and semantic-map model path.

Run:

```bash
cd PLMD-IIN
python main_INN.py -d ./data/IIN_EXP/
```

## MRON Eval

Follow `PLMD-MRON/README.md` to install the MRON environment.

Replace `PLMD-MRON/configs/local.yml` lines 44 and 45 with the obstacle-map model path and semantic-map model path.

Run:

```bash
cd PLMD-MRON
python main_MRON.py -d ./MRON_EXP/ \
    --num_agents 2 \
    --visualize 2 \
    --num_sem_categories 16 \
    --not_explore 1 \
    --task_config tasks/multi_objectnav_hm3d.yaml
```

## Citation

If you find PLMD useful, please cite our ICML 2026 paper:

```bibtex
@inproceedings{plmd2026,
  title     = {Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on the Habitat ecosystem and prior navigation repositories used for ObjectNav, Instance ImageNav, and multi-robot navigation research. Please also follow the licensing and citation requirements of the upstream projects and datasets used in your experiments.
