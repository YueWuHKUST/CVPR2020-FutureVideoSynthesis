CUDA_VISIBLE_DEVICES=0 python test_myback.py \
  --gpu_ids 0 \
  --batchSize 1 \
  --name cityscapes_test \
  --ngf 32 \
  --loadSize 1024 \
  --use_my_back \
  --ImagesRoot "/disk1/yue/cityscapes/leftImg8bit_sequence_512p/" \
  --npy_dir "/disk2/yue/final_result/test/" \
  --load_pretrain "./checkpoints/cityscapes/"
