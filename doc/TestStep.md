# Step 1 Generate mask

```
use ./dynamic/scripts/test.sh to generate dynamic masks, which indicates the moving cars

use ./process_scripts/gen_nonrigid_small/non_rigid.py to generate non rigid masks(including person and biker)

use ./process_scripts/gen_nonrigid_small/small.py to generate small object masks.
```

# Step 2 Background prediction

```
./back/scripts/street/test.sh
```

# Step 3 Background Inpainting

We use the [Generative Inpainting](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0) to inpaint missing region

# Step 4 Dynamic Motion Prediction

```
use the python files
./process_scripts/test_cityscapes.py ./process_scripts/test_kitti.py
to generate list storing the paths for testing

Then run
./fore/script/test_city.sh
```

# Step 5 Compute Occlusion Map

```
use the script ./process_scripts/occ.py

```

# Step 6 Video Inpainting

We use the [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting) to inpaint occlusion area.


Please note that, for Cityscapes dataset, we run the test procedure twice. After finishing the prediction of next 5 frames, we run the [semantic segmenation](https://github.com/NVIDIA/semantic-segmentation) method on predicted frames to obtain their semantic maps. It is used for next 5 to 10 frames prediction
