CUDA_VISIBLE_DEVICES=1 python test.py --gpu_ids 0 \
  --netG 'resnet' \
  --name 'cityscapes_final' \
  --dataset 'kitti' \
  --load_pretrain "./checkpoints/" \
  --ImagesRoot "/disk1/yue/kitti/raw_data/" \
  --SemanticRoot "/disk1/yue/kitti/semantic/" \
  --InstanceRoot "/disk1/yue/kitti/instance/" \
  --results_dir "kitti_val" \
  --phase "val"
  
