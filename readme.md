# Introduction

Hyperspectral images (HSIs) provide rich spectral
information that has been widely used in numerous computer vision
tasks. However, their low spatial resolution often prevents their
use in applications such as image segmentation and recognition.
Fusing low-resolution HSIs with high-resolution RGB images to
reconstruct high-resolution HSIs has attracted great research attention recently. In this paper, we propose an **unsupervised blind**
fusion network that operates on a **single** HSI and RGB image
pair and requires **neither** known degradation models nor any
training data. Our method takes full advantage of an unrolling
network and coordinate encoding to provide a state-of-the-art HSI
reconstruction. It can also **estimate the degradation parameters**
relatively accurately through the neural representation and implicit
regularization of the degradation model. The experimental results
demonstrate the effectiveness of our method both in simulations
and in our **real world** experiments.

[paper](https://ieeexplore.ieee.org/document/10037221)

# Start Fusion

## Parameters you might want to change

`--flag`: The image filename, used for reading image and the filename of the output file (e.g. log file).

`--dataset`: ('cave' or 'real' or any others) We used to differentiate the dataset, due to different dataset organization, e.g. some of them are Matlab files using dictionary data structure ({key}: {value}). You might need to debug the code to make some changes for your dataset.

`--model_dir`: The path to save models and outputs.

`--ker_sz`: The PSF kernel size, typically 32 or 8 for synthetic dataset and we set 6 for our real world scenario. See paper IV.A) and D). Note that for two synthetic dataset we used, the HR-RGB and LR-HSI are prepared online (see dataset_pre.py Class Dataset_pre()).

`--imsz`: HR-RGB size, typically 512. Change it to match your dataset. The LR-HSI size should be `--imsz` / `--ker_sz`.

Variables `tv_wei` and `lowrank_wei` are in the code (train.py), search for it. They are the penalty coefficient of the smooth term for SSF and PSF respectively. If the estimated SSF and PSF don't look right, try to strengthen or loosen the smooth constraint.

Check the rest parameters in train.py start from line 10.

## Fusion of a single image pair

`python train.py --model_dir=test/ --flag=??` and the others mentioned above.

### use pretrained model to continue a fusion

*pretrained model was trained on image fake_and_real_peppers_ms from CAVE dataset. And It is not trained to converge.*

`python train.py --model_dir=test/ --start_epoch=499 --end_epoch=500`

The above one only executes for one epoch, which gives you the results at epoch 500. For better performance, you can set `--end_epoch=` to a larger number, like `1000`.

# Real world dataset

[Google Drive](https://drive.google.com/file/d/1RKdqJQ_u_UhK9KCiw8st3gsv6oFSVY9A/view?usp=drive_link)

The real image dataset consists of 142 paired low-resolution
HSIs and trichromatic images. The trichromatic images were
captured by a HUAWEI P30Pro RYYB camera and a HUAWEI
P20 RGGB camera with the spatial resolution 5472×7296, and
the HSIs were captured by a compact scanning-based hyperspectral camera Specim IQ with spatial resolution 512×512 and
204 bands ranging from 400 nm to 1000 nm. For more information, see paper IV.A) and D).

# Contact

Leave an issue here or contact with me JiabaoL6 [at] uci.edu, make sure your title contains the keyword [fusion] so I can filter them.

# Cite us

If you find this work helpful, please don't forget to cite us, thank you :)

BibTeX

```
@ARTICLE{10037221,
  author={Li, Jiabao and Li, Yuqi and Wang, Chong and Ye, Xulun and Heidrich, Wolfgang},
  journal={IEEE Transactions on Computational Imaging}, 
  title={BUSIFusion: Blind Unsupervised Single Image Fusion of Hyperspectral and RGB Images}, 
  year={2023},
  volume={9},
  number={},
  pages={94-105},
  keywords={Degradation;Training;Image reconstruction;Data models;Optimization;Tensors;Hyperspectral imaging;Unsupervised Image Fusion;Blind Fusion;Hyperspectral Image Fusion},
  doi={10.1109/TCI.2023.3241549}}
```
or text
```
J. Li, Y. Li, C. Wang, X. Ye and W. Heidrich, "BUSIFusion: Blind Unsupervised Single Image Fusion of Hyperspectral and RGB Images," in IEEE Transactions on Computational Imaging, vol. 9, pp. 94-105, 2023, doi: 10.1109/TCI.2023.3241549.
keywords: {Degradation;Training;Image reconstruction;Data models;Optimization;Tensors;Hyperspectral imaging;Unsupervised Image Fusion;Blind Fusion;Hyperspectral Image Fusion},

```

