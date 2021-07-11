CUDA_VISIBLE_DEVICES=0 python test_myback.py \
  --gpu_ids 0 \
  --batchSize 1 \
  --name cityscapes_test \
  --ngf 32 \
  --loadSize 1024 \
  --use_my_back \
  --ImagesRoot "./data/cityscapes/leftImg8bit_sequence_512p/" \
  --npy_dir "./process_scripts/test/" \
  --load_pretrain "./checkpoints/cityscapes/"
