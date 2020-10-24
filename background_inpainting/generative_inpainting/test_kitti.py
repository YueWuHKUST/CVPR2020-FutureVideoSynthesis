import argparse

import cv2, os, time
import numpy as np
import tensorflow as tf
import neuralgym.neuralgym as ng
from inpaint_model import InpaintCAModel
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='./ckpt/20191102072444392346_cqcpu2_kitti_NORMAL_wgan_gp_256p/', type=str,
                    help='The directory of tensorflow checkpoint.')

#20190916064133972067_cqcpu2_kitti_NORMAL_wgan_gp_256p

image_root = "/disk1/yue/kitti/raw_data/val/"
semantic_root = "/disk1/yue/kitti/semantic/val/gray/"
save_dir = "/disk1/yue/kitti/inpainting/val/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

remain_list = ['2011_09_28_drive_0001_sync', \
                '2011_09_28_drive_0002_sync', \
                '2011_09_29_drive_0026_sync',\
                '2011_09_29_drive_0071_sync']


def load_all_image_paths(image_root, semantic_root):
    #Image dir is Dir in "train/val/test/" level
    city_dir = os.listdir(image_root)
    city_dir.sort()
    video = []
    for i in range(len(city_dir)):
        #if city_dir[i] not in remain_list:
        #    continue
        # image_dir_1 is Dir in "aachen" level
        image_dir_1 = image_root + city_dir[i] + "/image_02/data/"
        semantic_dir_1 = semantic_root + city_dir[i] + "/image_02/data_transformed/"
        frame_list = os.listdir(image_dir_1)
        frame_list.sort()
        for k in range(len(frame_list)):
            image_full_path = image_dir_1 + frame_list[k]
       	    semantic_full_path = semantic_dir_1 + frame_list[k]
            #print(semantic_full_path)
            assert os.path.exists(image_full_path)
            assert os.path.exists(semantic_full_path)
            video.append((image_full_path, semantic_full_path, city_dir[i], frame_list[k]))
    return video



sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

def clip_mask(mask):
    mask = mask > 0.5
    return mask.astype(np.int32)




def mask_remove_shadow(mask):
    # 1-background 0-inpainted shape 256x512
    h, w, _ = mask.shape
    for i in range(w):
        curr_colwn = mask[:,i,0]
        idy = np.where(curr_colwn == 0)[0]
        #print(idy)
        if len(idy) > 0:
            max_idy = idy.max()
            mask[max_idy:np.minimum(max_idy + 15,256),i,:] = 0
    return mask


dilate_size = 3
iteration_num = 1
if __name__ == "__main__":
    #ng.get_gpus(1)
    args = parser.parse_args()
    height = 256
    width = 832
    all_paths = load_all_image_paths(image_root, semantic_root)
    model = InpaintCAModel()
    sess = tf.Session(config=sess_config)
    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, height, width*2, 3))
    output = model.build_server_graph(input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
    #root = "/disk2/yue/VideoPred/new_w_3/result/kitti_11.14/test_latest/"
    #root = "/disk2/yue/VideoPred/new_w_3/result/kitti_11.14_less_objects/test_latest/"
    root = "/disk2/yue/server6_backup/final/finetune_0.002_add_person/kitti/"
    #root = "/disk2/yue/VideoPred/new_w_3/result/ablation1/test_latest/"
    #root = "/disk2/yue/VideoPred/new_w_3/result/kitti_ngf64_no_fm_pred3_add_D_T/test_latest/"
    #root = "/disk2/yue/VideoPred/result/kitti/kitti_ngf64_no_fm_pred3_add_D_T/test_latest/"
    for ind in range(2000):
        #if ind % 10 != 0:
        #    continue
        print("Processing Ind %d" % (ind))
        st = time.time()
        index = ind
        save_dir = root + "%04d/"%(ind)
        for frame_id in range(4):
            image = cv2.imread(save_dir + "input_image_%02d.png"%frame_id)
            mask = np.array(Image.open(save_dir + "input_mask_%02d.png"%frame_id).convert('RGB'))
            mask = 255 - mask
            kernel = np.ones((dilate_size,dilate_size),np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=iteration_num)
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid,:]
            #print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            cv2.imwrite(save_dir + "input_image_inpainted_%02d.png"%frame_id, result[0][:, :, ::-1])

        for frame_id in range(6):
            image = cv2.imread(save_dir + "warp_image_bwd_%02d.png"%frame_id)
            h, w, _ = image.shape
            if h != 256:
                image = cv2.resize(image, (832, 256))
            mask = Image.open(save_dir + "warp_mask_bwd_%02d.png"%frame_id)
            w_s, h_s = mask.size
            if h_s != 256:
                mask = mask.resize((832, 256), Image.NEAREST)
            mask = np.array(mask.convert('RGB'))
            mask = 255 - mask
            kernel = np.ones((dilate_size,dilate_size),np.uint8)
            mask = cv2.dilate(mask, kernel,iterations=iteration_num)
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid,:]
            #print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            cv2.imwrite(save_dir + "warp_image_bwd_inpainted_%02d.png"%frame_id, result[0][:, :, ::-1])
    sess.close()









