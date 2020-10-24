CUDA_VISIBLE_DEVICES=7 python tools/video_inpaint_kitti.py \
        --FlowNet2 \
        --DFC \
        --frame_dir ./demo/frames \
        --MASK_ROOT ./demo/masks \
        --ResNet101 \
        --img_size 256 832 \
        --Propagation \
        --enlarge_mask
         #\
        #--th_warp 10
