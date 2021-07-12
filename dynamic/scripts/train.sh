python train.py \
  --gpu_ids 0 \
  --netG 'resnet' \
  --name 'dynamic' \
  --dataset 'kitti' \
  --niter 20 \
  --ngf 64 \
  --niter_decay 20 \
  --load_pretrain "./checkpoints/city/"
  
