CUDA_VISIBLE_DEVICES=1 python tools/video_inpaint_city.py \
        --FlowNet2 \
        --DFC \
        --frame_dir ./demo/frames \
        --MASK_ROOT ./demo/masks \
        --ResNet101 \
        --img_size 512 1024 \
        --Propagation \
        --enlarge_mask
         #\
        #--th_warp 10
