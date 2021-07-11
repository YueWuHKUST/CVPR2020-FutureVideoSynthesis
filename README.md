# Official implementation of Paper "Future Video Synthesis With Object Motion Prediction"(CVPR 2020)

# Enviroment

Python 3

Pytorch1.0.0

# To do
I plan to clean up the code in one month


# Components

There are exists several components in our framework. This repo only contain modified files for [Generative Inpainting](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0) and [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting).

We use [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) to compute optical flow. Please put the code under './*/models/' directory

- Dynamic Object Detection

- Background Prediction

- Background Inpainting following [Generative Inpainting](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0)

- Dynamic Object Motion Prediction

- Video Inpainting modified from [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)

# Data preparation
This [document](https://github.com/YueWuHKUST/FutureVideoSynthesis/blob/main/doc/Data_preparation.md) illustrates how the data preprocessing is done. I release the test result in next section. If you want the preprocessed **test dataset** to verify the result, please drop me an email.

```bash
data
├── Cityscapes
│   ├── depth
│   ├── dynamic
│   ├── for_val_data
│   ├── instance_upsnet
│   │   └── origin_data
│   │   └── val
│   ├── leftImg8bit_sequence_512p (Citysapes images in 512x1024)
│   │   └── val
│   │   │     └── frankfurt
│   │   │     └── lindau
│   │   │     └── munster
│   ├── non_rigid_mask
│   ├── semantic 
│   ├── small_object_mask
├── Kitti
│   ├── depth
│   ├── dynamic
│   ├── for_val_data
│   ├── instance_upsnet
│   │   └── origin_data
│   │   └── val
│   ├── raw_data_56p (Kitti images in 256x832)
│   │   └── val
│   │   │    └── 2011_09_26_drive_0060_sync
│   │   │    │    └── image_02
│   │   │    │    │     └── data
│   │   │    └── 2011_09_26_drive_0084_sync
│   │   │    └── 2011_09_26_drive_0093_sync
│   │   │    └── 2011_09_26_drive_0096_sync
│   ├── non_rigid_mask
│   ├── semantic 
│   ├── small_object_mask
```



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

# Acknowledgement
The code is developed based on Vid2Vid https://github.com/NVIDIA/vid2vid
