# Building Non-uniform Degradation Model for Position-aware Hyperspectral Image Fusion (Accepted by TPAMI, to appear)
### Jie Lian [![](https://img.shields.io/badge/google-scholar-yellow)](https://scholar.google.com.hk/citations?hl=zh-CN&user=WcyOmdQAAAAJ), Lizhi Wang [![](https://img.shields.io/badge/google-scholar-yellow)](https://scholar.google.com.hk/citations?hl=zh-CN&user=FEprmwYAAAAJ), Lin Zhu [![](https://img.shields.io/badge/google-scholar-yellow)](https://scholar.google.com.hk/citations?hl=zh-CN&user=32d6xfEAAAAJ), Renwei Dian [![](https://img.shields.io/badge/google-scholar-yellow)](https://scholar.google.com.hk/citations?hl=zh-CN&user=EoTrH5UAAAAJ), Zhiwei Xiong [![](https://img.shields.io/badge/google-scholar-yellow)](https://scholar.google.com.hk/citations?hl=zh-CN&user=Snl0HPEAAAAJ), Hua Huang [![](https://img.shields.io/badge/google-scholar-yellow)](https://scholar.google.com.hk/citations?hl=zh-CN&user=EplUB7oAAAAJ&view_op=list_works&sortby=pubdate)

<hr />

> **Abstract:** *The fusion of low-spatial-resolution hyperspectral image (LR-HSI) with high-spatial-resolution multispectral image (HR-MSI) has become an effective way to obtain the high-spatial-resolution hyperspectral image (HR-HSI). Currently, learning-based methods have emerged as the mainstream solution in this field. However, these methods typically rely on predefined or simplified degradation models during fusion training, resulting in inaccurate supervision of the fusion networks. Meanwhile, most methods overlook the degradation characteristics in designing the fusion networks, leading to a mismatch between the degradation and fusion processes. These limitations ultimately result in unsatisfactory fusion performance on real data. To enhance the practicality of learning-based methods, accurate degradation modeling and effective network design have become the critical priorities. We observe that, in practical scenarios, the degree of pixel degradation varies across different positions due to the unforeseen factors such as illumination variations and imaging system fluctuations. Considering this, we propose a non-uniform degradation model (NUD), which introduces non-uniformity into the degradation processes of LR-HSI and HR-MSI. In addition, we emphasize that the essence of fusion is to reverse the degradation process. Therefore, to align with the non-uniform degradation process, the fusion process should exhibit similar positional specificity. For this purpose, we propose a position-aware fusion network (PAF), which employs positional encoding to endow the fusion process with the position-aware attribute. Experimental results show that our proposed methods provide an effective solution for HSI fusion in practical scenarios.*

<hr />

## Network Architecture
### Non-uniform Degradation Model
<img src="fig/NUD.png" width=600 height=375>

### Position-aware Fusion Network
<img src="fig/PAF.png" width=600 height=375>

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
-- hypsen.mat (Real dataset captured in practical scenarios.)
-- paviaU.mat (Auxiliary dataset used in simulation experiments.)

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

The paper has been accepted and is currently awaiting publication. If this repo helps you, please consider citing our work:

```shell
@article{nud_paf,
  title={Building Non-uniform Degradation Model for Position-aware Hyperspectral Image Fusion},
  author={Jie Lian and Lizhi Wang and Lin Zhu and Renwei Dian and Zhiwei Xiong and Hua Huang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```

## 5. Contact:

For any question, please contact:

```shell

lianjie@bit.edu.cn

```

