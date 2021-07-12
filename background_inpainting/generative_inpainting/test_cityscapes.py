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
parser.add_argument('--checkpoint_dir', default='./ckpt/20191030054240327253_cqcpu2_city_512p_NORMAL_wgan_gp_HD/', type=str,
                    help='The directory of tensorflow checkpoint.')


sp = 1024
height = 512
width = 1024
ImagesRoot = "./data/cityscapes/leftImg8bit_sequence_512p/val/"
SemanticRoot = "./data/cityscapes/semantic/val/"
def load_all_image_paths(image_root, semantic_root):
    #Image dir is Dir in "train/val/test/" level
    city_dir = os.listdir(image_root)
    city_dir.sort()
    video = []
    for i in range(len(city_dir)):
        # image_dir_1 is Dir in "aachen" level
        image_dir_1 = image_root + city_dir[i]
        semantic_dir_1 = semantic_root + city_dir[i] + "/gray"
        frame_list = os.listdir(image_dir_1)
        frame_list.sort()
        for j in range(len(frame_list)//30):
            image = []
            semantic = []
            for k in range(j*30 + 0, (j+1)*30):
                # Append image paths and assert whether exists
                full_image_path = image_dir_1 + "/" + frame_list[k]
                assert os.path.isfile(full_image_path)
                image.append(full_image_path)

                # Append semantic paths and assert whether exists
                full_semantic_path = semantic_dir_1 + "/" + frame_list[k]
                assert os.path.isfile(full_semantic_path)
                semantic.append(full_semantic_path)

            video.append((image, semantic))
    return video



sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True



def mask_remove_shadow(mask):
    # 1-background 0-inpainted shape 256x512
    h, w,_ = mask.shape
    for i in range(w):
        curr_colwn = mask[:,i,0]
        idy = np.where(curr_colwn == 0)[0]
        #print(idy)
        if len(idy) > 0:
            max_idy = idy.max()
            mask[max_idy:np.minimum(max_idy + 15,256),i,:] = 0
    return mask

def S2mask(semantic):
    print(semantic.shape)
    h, w = semantic.shape
    mask = np.zeros((h, w))
    for i in range(11,19):
        mask[semantic == i] = 1
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, [1,1,3])
    return mask


if __name__ == "__main__":
    #ng.get_gpus(1)
    args = parser.parse_args()

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
    root = "./result/cityscapes/"
    for ind in range(500):
        print("Processing Ind %d" % (ind))
        st = time.time()
        index = ind
        save_dir = root + "%04d/"%(ind)
        for frame_id in range(4):
            image = cv2.imread(save_dir + "input_image_%02d.png"%frame_id)
            mask = np.array(Image.open(save_dir + "input_mask_%02d.png"%frame_id).convert('RGB'))
            mask = 255 - mask
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.dilate(mask, kernel)
            #mask = S2mask(semantic)
            #if mask.sum() == 0:
            #    cv2.imwrite(save_dir + "pred_image_inpainted_%02d.png"%frame_id, image)
            #    continue
            #mask = mask_remove_shadow(mask)
            #mask = 255 - mask*255
            #mask = mask*255
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid,:]
            #print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            #cv2.imwrite(save_dir + "Predict_fwd_warp_v2_%02d.png"%frame_id, result[0][:, :, ::-1])
            #cv2.imwrite(save_dir + "input_image_inpainted_%02d.png"%frame_id, result[0][:, :, ::-1])

        for frame_id in range(10):
            image = cv2.imread(save_dir + "warp_image_bwd_%02d.png"%frame_id)
            mask = np.array(Image.open(save_dir + "warp_mask_bwd_%02d.png"%frame_id).convert('RGB'))
            mask = 255 - mask
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.dilate(mask, kernel)
            #mask = S2mask(semantic)
            #if mask.sum() == 0:
            #    cv2.imwrite(save_dir + "pred_image_inpainted_%02d.png"%frame_id, image)
            #    continue
            #mask = mask_remove_shadow(mask)
            #mask = 255 - mask*255
            #mask = mask*255
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid,:]
            #print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            #cv2.imwrite(save_dir + "Predict_fwd_warp_v2_%02d.png"%frame_id, result[0][:, :, ::-1])
            #cv2.imwrite(save_dir + "warp_image_bwd_inpainted_%02d.png"%frame_id, result[0][:, :, ::-1])
    sess.close()









