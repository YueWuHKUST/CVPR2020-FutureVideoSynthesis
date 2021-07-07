# Instance map generation

For instance map generation, we adopt UPSNet to generate instance maps: https://github.com/uber-research/UPSNet

We slightly modified their code, remove result within low confidence score. And we detect the catergories including ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'] The code we use is https://github.com/YueWuHKUST/UPSNet

The test step is:
1. Call the creat_json.py file to generate json files containing the paths of imgs to be tested.
2. Edit the json path in https://github.com/YueWuHKUST/UPSNet/blob/master/upsnet/experiments/upsnet_resnet101_cityscapes_w_coco_16gpu.yaml#L14
3. Calling CUDA_VISIBLE_DEVICES=0 python upsnet/upsnet_end2end_test_kitti.py --cfg upsnet/experiments/upsnet_resnet101_kitti_w_coco_16gpu.yaml --weight_path ./model/upsnet_resnet_101_cityscapes_w_coco_3000.pth to generate instance maps.

# Semantic map generation

Our semantic map is generated using https://github.com/NVIDIA/semantic-segmentation. The test step is the same with their official code.

# Depth map generation

Since we predict the future of each instance separately, we need their relative depth to determine their order when combining them together. Since we only predict next 0.58 seconds in Cityscapes, next 0.5 seconds in Kitti, we assume this relative order is the same as the last input frame. Thus, we use https://github.com/yzcjtr/GeoNet to compute the depth map for last input frame, and the test step is the same with the official code.




