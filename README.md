# Official implementation of Paper "Future Video Synthesis With Object Motion Prediction"(CVPR 2020)

# Enviroment

Python 3

Pytorch1.0.0

# Components

There are exists several components in our framework

- Dynamic Object Detection

- Background Prediction

- Background Inpainting following [Generative Inpainting](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0)

- Dynamic Object Motion Prediction

- Video Inpainting modified from [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)


# Pretrained Models

The Dynamic Detection Model weight is in [Google Drive]().
|                  | Cityscapes     | Kitti |
| ------           | ------------    | ------|
| Background prediction | [Google Drive]() | [Google Drive]()|
| Background Inpainting | [Google Drive]() | [Google Drive]()|
| Dynamic Object Prediction | [Google Drive]() | [Google Drive]()|

# Results
All of our test results are stored in [Google Drive]()
The details of test setting can be found in [link]()

If you want to test the model from beginning, the precise test step is in [link]()

# Citation
If you use our code or paper, please cite:
```
@InProceedings{Yue_2020_CVPR,
author = {Yue Wu, Rongrong Gao, Jaesik Park and Qifeng Chen},
title = {Future Video Synthesis with Object Motion Prediction},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
