# Cityscapes

We use their provided validation set as test set. The test set contains 500 videos. We use the first 4 frame in each sequence as input, and predict the next 10 frames. 


# KITTI

The training split we use is:
```
2011_09_26_drive_0001_sync  2011_09_26_drive_0018_sync  2011_09_26_drive_0104_sync
2011_09_26_drive_0002_sync  2011_09_26_drive_0048_sync  2011_09_26_drive_0106_sync
2011_09_26_drive_0005_sync  2011_09_26_drive_0051_sync  2011_09_26_drive_0113_sync
2011_09_26_drive_0009_sync  2011_09_26_drive_0056_sync  2011_09_26_drive_0117_sync
2011_09_26_drive_0011_sync  2011_09_26_drive_0057_sync  2011_09_28_drive_0001_sync
2011_09_26_drive_0013_sync  2011_09_26_drive_0059_sync  2011_09_28_drive_0002_sync
2011_09_26_drive_0014_sync  2011_09_26_drive_0091_sync  2011_09_29_drive_0026_sync
2011_09_26_drive_0017_sync  2011_09_26_drive_0095_sync  2011_09_29_drive_0071_sync
```
The test split we use is:
```
2011_09_26_drive_0060_sync  2011_09_26_drive_0084_sync  2011_09_26_drive_0093_sync  2011_09_26_drive_0096_sync
```
The train/test split is random choosed

For testing, we overlapply sample sequence of 4 frames, and predict next 5 frames. The total sequences in testing set is 1337.
