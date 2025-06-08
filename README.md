# Equivariance Makes Volumetric Grasping Great Again

This repo also contains a clean and simple implementation for [GIGA](https://github.com/UT-Austin-RPL/GIGA) and [IGD](https://github.com/mousecpn/Implicit-Grasp-Diffusion), which is compatible for their pretrained checkpoint.

## Introduction


## Installation

1. Create a conda environment.

2. Install packages list in [requirements.txt](requirements.txt). Then install `torch-scatter` following [here](https://github.com/rusty1s/pytorch_scatter), based on `pytorch` version and `cuda` version.

3. Go to the root directory and install the project locally using `pip`

```
pip install -e .
```

4. Data collection can be referred to this [repo](https://github.com/mousecpn/grasp-data-collection).

## Training

### Train EquiGIGA

Run:

```bash
python train_equigiga.py --dataset /path/to/new/data --dataset_raw /path/to/raw/data
```

### Train EquiIGD

Run:

```bash
python train_equiigd.py --dataset /path/to/new/data --dataset_raw /path/to/raw/data
```

### Train GIGA

Run:

```bash
python train_giga.py --dataset /path/to/new/data --dataset_raw /path/to/raw/data
```

### Train IGD

Run:

```bash
python train_igd.py --dataset /path/to/new/data --dataset_raw /path/to/raw/data
```

## Validation

Run:

```bash
python scripts/sim_grasp_multiple.py --num-view 1 --object-set (packed/test | pile/test) --scene (packed | pile) --num-rounds 100 --sideview --add-noise dex --force --best --model /path/to/model --type (giga | igd | equi_giga | equi_igd) --result-path /path/to/result
```

This commands will run experiment with each seed specified in the arguments.

Run `python scripts/sim_grasp_multiple.py -h` to print a complete list of optional arguments.


## Citing


