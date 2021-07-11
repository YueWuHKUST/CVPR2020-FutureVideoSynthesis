CUDA_VISIBLE_DEVICES=7 python test.py --gpu_ids 0 \
  --netG 'resnet' \
  --name 'cityscapes_final' \
  --dataset 'cityscapes' \
  --load_pretrain "./checkpoints/" \
  --ImagesRoot "./data/cityscapes/leftImg8bit_sequence_512p/" \
  --SemanticRoot "./data/cityscapes/semantic/" \
  --InstanceRoot "./data/cityscapes/instance_upsnet/" \
  --results_dir "cityscapes_val" \
  --phase "val"
  # \
  #--how_many 4
  
