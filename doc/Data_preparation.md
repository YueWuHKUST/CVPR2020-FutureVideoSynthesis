# Instance map generation

For instance map generation, we adopt UPSNet to generate instance maps: https://github.com/uber-research/UPSNet

Cityscapes and Kitti dataset are both tested using following model
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet101_cityscapes_w_coco_16gpu.yaml --weight_path ./model/upsnet_resnet_101_cityscapes_w_coco_3000.pth

And we detect the catergories including ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# Semantic map generation

to be add


# Optical flow generation

to be add




