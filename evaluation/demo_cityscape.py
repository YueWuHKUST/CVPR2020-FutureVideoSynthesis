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
image_my = tf.placeholder(tf.float32)
distance_my = lpips_tf.lpips(image_0, image_my, model='net-lin', net='alex')
ssim_my = tf.image.ssim_multiscale(image_0[0,...], image_my[0,...], max_val=1.0)

## Load GT images
GT_root = "/disk1/yue/cityscapes/leftImg8bit_sequence_512p/val/"
city_dir = os.listdir(GT_root)
city_dir.sort()
true_videos = []
for i in range(len(city_dir)):
	frame_dir = GT_root + city_dir[i]
	frame_list = os.listdir(frame_dir)
	frame_list.sort()
	for j in range(len(frame_list)//30):
		image = []      
		for k in range(j*30, (j+1)*30):
			full_image_path = frame_dir + "/" + frame_list[k]
			assert os.path.isfile(full_image_path)
			image.append((full_image_path, city_dir[i], frame_list[k]))
		true_videos.append(image)


my_root = "/disk2/yue/final/final_result/cityscapes/"
my_videos = []
for i in range(500):
	sub_dir = my_root + "%04d/"%i
	images = []
	for k in range(10):
		images.append(sub_dir + "pred_%03d.png"%k)
	my_videos.append(images)


ipips_score_mine = np.zeros(10)
ssim_score_mine = np.zeros(10)


for i in range(500):
	#print("process ", i)
	for f in range(10):
		true_image = np.expand_dims(np.array(Image.open(true_videos[i][4 + f][0])), axis=0)/255
		my_image = np.expand_dims(np.array(Image.open(my_videos[i][f])), axis=0)/255
		ipips_m, ssim_m = sess.run([distance_my, ssim_my], feed_dict={
					image_0: true_image, \
					image_my: my_image, 
			})
		#print("ssim shape", ssim.shape)
		ipips_score_mine[f] += ipips_m
		ssim_score_mine[f] += ssim_m

	if i % 100 == 0:
		for f in range(10):
			print("ipips my %d= "%f, ipips_score_mine[f]/(i+1), "ssim %d= "%f, ssim_score_mine[f]/(i+1))

for f in range(10):
	print("ipips my %d= "%f, ipips_score_mine[f]/500, "ssim %d= "%f, ssim_score_mine[f]/500)
