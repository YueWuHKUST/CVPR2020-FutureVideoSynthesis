# Official implementation of Paper "Future Video Synthesis With Object Motion Prediction"(CVPR 2020)

# Enviroment

Python 3

Pytorch1.0.0

# Components

There are exists several components in our framework. This repo only contain modified files for [Generative Inpainting](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0) and [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting).

We use [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) to compute optical flow. Please put the code under './*/models/' directory

- Dynamic Object Detection

- Background Prediction

- Background Inpainting following [Generative Inpainting](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0)

- Dynamic Object Motion Prediction

- Video Inpainting modified from [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)


# Pretrained Models and Results

The **test results** and pretrained models are in [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ywudg_connect_ust_hk/ErucIeTSpbNCn7Sf5xB24F0BbSqrcRFuicNPZgK_3TXcDg?e=TBKfgU)

The details of test setting can be found in [link](https://github.com/YueWuHKUST/FutureVideoSynthesis/blob/main/doc/TestSetting.md)

If you want to test the model from beginning, the precise test step is in [link](https://github.com/YueWuHKUST/FutureVideoSynthesis/blob/main/doc/TestStep.md)

# Evaluation
We use [LPIPS](https://github.com/alexlee-gk/lpips-tensorflow) for evaluation


# Citation
If you use our code or paper, please cite:
```
@InProceedings{Yue_2020_CVPR,
author = {Yue Wu and Rongrong Gao and Jaesik Park and Qifeng Chen},
title = {Future Video Synthesis with Object Motion Prediction},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```


# Contact
If you have any question, please feel free to contact me (Yue Wu, ywudg@connect.ust.hk)

