CUDA_VISIBLE_DEVICES=7 python test.py --gpu_ids 0 \
  --netG 'resnet' \
  --name 'cityscapes_final' \
  --dataset 'cityscapes' \
  --load_pretrain "./checkpoints/" \
  --ImagesRoot "/disk1/yue/cityscapes/leftImg8bit_sequence_512p/" \
  --SemanticRoot "/disk1/yue/cityscapes/semantic_new/" \
  --InstanceRoot "/disk1/yue/cityscapes/instance_upsnet/" \
  --results_dir "cityscapes_val" \
  --phase "val"
  # \
  #--how_many 4
  
