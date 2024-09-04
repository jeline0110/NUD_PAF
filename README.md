# NUD_PAF

This is an official implementation of **Building Non-uniform Degradation Model for Position-aware Hyperspectral Image Fusion**.

## 1. Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- pytorch
- numpy
- scikit-image
- scipy
- seaborn
- matplotlib

## 2. Datasets:

Datasets are available in `/datasets`.
```shell
hypsen.mat (Real dataset captured in practical scenarios.)
paviaU.mat (Auxiliary dataset used in simulation experiments.)

```

## 3. Experiments:

- Degradation estimation on real data:
```shell
cd NUD/
python main_deg.py 
```

- Fusion on real data:
```shell
cd PAF/
python main_fus_real.py 
```

- Fusion on synthetic data:
```shell
cd PAF/
python main_fus_syn.py 
```

The results will be output into `results/`. 

## 4. Citation:

If this repo helps you, please consider citing our work:

```shell
The paper has not been published yet, please wait patiently for the publication.
```

## 5. Contact:

For any question, please contact:

```shell

lianjie@bit.edu.cn

```

