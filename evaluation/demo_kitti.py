import numpy as np 
import tensorflow as tf
import lpips_tf
import os
from PIL import Image
import glob

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
image_0 = tf.placeholder(tf.float32)
image_1 = tf.placeholder(tf.float32)
distance = lpips_tf.lpips(image_0, image_1, model='net-lin', net='alex')
msssim = tf.image.ssim_multiscale(image_0[0,...], image_1[0,...], max_val=1)

## Load GT images
GT_root = "./data/kitti/raw_data_256p/val/"
city_dir = os.listdir(GT_root)
city_dir.sort()
true_videos = []
for i in range(len(city_dir)):
    frame_dir = GT_root + city_dir[i] + "/image_02/data/"
    frame_list = os.listdir(frame_dir)
    frame_list.sort()
    for j in range(len(frame_list)-8):
        image = []
        for k in range(j,j+9):
            full_image_path = frame_dir + "/" + frame_list[k]        
            image.append((full_image_path, city_dir[i], frame_list[k]))
        true_videos.append(image)
assert len(true_videos) == 1337


my_root = "./result/kitti/"
my_videos = []
for i in range(1337):
	sub_dir = my_root + "%04d/"%(i)
	images = []
	for k in range(5):
		path = sub_dir + "pred_%03d.png"%k
		images.append(path)
	my_videos.append(images)








ipips_score_mine = np.zeros(5)
ssim_score_mine = np.zeros(5)
msssim_score_mine = np.zeros(5)
psnr_score_mine = np.zeros(5)

sp = 256
cnt = 0
print("true_image_len = ", len(true_videos))

num = 1337

for i in range(num):
	for f in range(5):
		true_image = np.expand_dims(np.array(Image.open(true_videos[i][4 + f][0]))/255.0,axis=0)
		my_image = np.expand_dims(np.array(Image.open(my_videos[i][f]))/255.0,axis=0)

		ipips_v, msssim_v = sess.run([distance, msssim], feed_dict={
				image_0: true_image, \
				image_1: my_image
		})
		ipips_score_mine[f] += ipips_v
		msssim_score_mine[f] += msssim_v
	if i % 100 == 0:
		print("process ", i)
		for f in range(5):
			print("ipips mine    %d=%.5f"%(f, ipips_score_mine[f]/(i+1)),   "ssim %d=%.5f"%(f, msssim_score_mine[f]/(i+1)))

for f in range(5):			
	print("ipips mine    %d=%.5f"%(f, ipips_score_mine[f]/num),   "ssim %d=%.5f"%(f, msssim_score_mine[f]/num))
