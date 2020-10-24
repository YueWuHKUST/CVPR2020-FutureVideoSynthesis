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
|        | Cityscapes   | Kitti |
| ------ | ------------ | ------|
| Dynamic Detection| [Google Drive]()|
| Background prediction | [Google Drive]() | [Google Drive]()|
| Background Inpainting | [Google Drive]() | [Google Drive]()|
| Dynamic Object Prediction | [Google Drive]() | [Google Drive]()|

# Results

All of our test results are stored in [Google Drive]()

# Testing 

1. Running Dynamic Detection Model to determine which objects are moving
2. Generate Background Prediction Result 
3. Inpaint the predicted Background
4. Generate Dynamic Object Motion Prediction
5. Composite the foreground and background together
6. Running the Video Inpainting Method to obtain full results





