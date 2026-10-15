# Equivariant Volumetric Grasping

This repo is the official implementation of the paper: Equivariant Volumetric Grasping.

[[Project](https://mousecpn.github.io/evg-page/)] [[Arxiv](https://arxiv.org/abs/2507.18847)] [[Realworld](https://github.com/mousecpn/fr3_grasp_ws)]

<p align="center">
    <img src="asset/equi_demo.png" alt="Equivariance" width="600">
</p>

## :star2: Highlights

- An [Equivariant Triplane UNet](model/triunet_induced.py) for volumetric grasping

- We adapt Equivariant Triplane UNet on [EquiGIGA](model/equi_giga.py) and [EquiIGD](model/equi_igd.py) models.

- This repo also contains a clean and simple implementation for GIGA and IGD, which is compatible with their pretrained checkpoint.

- The current repo implements a full equivariant version of triplane representation. The previous version can be found in the branch: backup/pre-induced-migration-20260930.

## :bulb: Key insight

<p align="center">
    <img src="asset/triequi.gif" alt="insight" width="1000">
</p>


## 🔧 Installation

1. Create a conda environment.

2. Install packages list in [requirements.txt](requirements.txt). Then install `torch-scatter` following [here](https://github.com/rusty1s/pytorch_scatter) (only for GIGA and IGD), based on `pytorch` version and `cuda` version.


3. Data collection can be referred to this [repo](https://github.com/mousecpn/grasp-data-collection).

## 📈 Training

```bash
./train.sh (equi_giga | equi_igd) --dataset /path/to/new/data --dataset_raw /path/to/raw/data --num-workers 12 --epochs 12 --batch-size 128
```

## 🎯 Validation

```bash
python sim_grasp_multiple.py --num-view 1 --object-set (packed/test | pile/test) --scene (packed | pile) --num-rounds 100 --sideview --add-noise dex --force --best --model /path/to/model --type (equi_giga | equi_igd) --result-path /path/to/result
```

This commands will run experiment with each seed specified in the arguments.

Run `python sim_grasp_multiple.py -h` to print a complete list of optional arguments.

## 🙏 Acknowledgement
- Data collection code from [GIGA](https://github.com/UT-Austin-RPL/GIGA) and [OrbitGrasp](https://github.com/BoceHu/orbitgrasp).
- Triplane UNet and Decoder are largely based on [GIGA](https://github.com/UT-Austin-RPL/GIGA).
- Diffusion rotation decoders are from [IGD](https://github.com/mousecpn/Implicit-Grasp-Diffusion).
- Thanks to [ESCNN](https://github.com/QUVA-Lab/escnn) for their easy-to-use steerable convolution coding tutorials and tools.


## Citing
```
@article{song2025equivariant,
  title={Equivariant Volumetric Grasping},
  author={Song, Pinhao and Hu, Yutong and Li, Pengteng and Detry, Renaud},
  journal={arXiv preprint arXiv:2507.18847},
  year={2025}
}
```

