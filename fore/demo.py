import numpy as np
import scipy
import os
from PIL import Image
import glob
import pandas as pd

def read_image(path, h=None, w=None):
    return np.array(Image.open(path))






def check_valid_car_mask(frames, car_masks):
    flag = True
    #for p in range(9):
    # Check first frame
    o = np.array(Image.open(frames[0]).resize((512,256), Image.NEAREST))/255
    car = np.array(Image.open(car_masks[0]).resize((512, 256), Image.NEAREST))/255
    if np.sum(car*o) / np.sum(o) < 0.5:
        flag = False
        return flag
    return flag


def load_tracked_dict(path, images, car_masks, semantics, backs, train_list):

    for k in range(21):
        frame_dir = path + "%02d/"%k
        if not os.path.exists(frame_dir):
            continue
        object_list = os.listdir(frame_dir)
        object_list.sort()
        object_ret = []
        for o in range(len(object_list)):
            object_dir = frame_dir + object_list[o] + "/"
            #print(object_dir)
            frames = glob.glob(object_dir + "*.png")
            #print(frames)
            frames.sort()
            flag = check_valid_car_mask(frames, car_masks[k:k+9])
            if flag is True:
                train_list.append((images[k:k+9], semantics[k:k+9], backs[k:k+9], frames))
    return train_list


def load_tracked_dict_val(path):
    frames_list = os.listdir(path)
    frames_list.sort()
    #print("frames_list = ", frames_list)
    #if len(frames_list) != 21:
    #    return None
    ret = [None]*1
    for k in range(len(frames_list)):
        frame_dir = path + frames_list[k] + "/"
        #print("frame_dir = ", frame_dir)
        object_list = os.listdir(frame_dir)
        object_list.sort()
        object_ret = []
        #print("object_list = ", object_list)
        for o in range(len(object_list)):
            object_dir = frame_dir + object_list[o] + "/"
            frames = glob.glob(object_dir + "*.png")
            frames.sort()
            object_ret.append(frames)
        ret[k] = object_ret
    #print(ret)
    return ret

def load_all_image_paths_track_val(opt, phase):
    image_dir = opt.ImagesRoot + phase + '/'
    inpainted_back_dir = opt.BackRoot + phase + '/'
    instance_gt_dir = opt.InstanceGTRoot + phase + '/'
    instance_mask_rcnn_dir = opt.Instance_maskrcnn + phase + '/'
    semantic_psp_dir = opt.SemanticRoot + phase + '/'
    semantic_gt_dir = opt.SemanticGTRoot + phase + '/'
    depth_dir = opt.DepthMapRoot + phase + '/'
    
    track_instance_dir = opt.TrackInstanceRoot
    city_dir = os.listdir(image_dir)
    city_dir.sort()
    cnt = 0
    video = []
    for i in range(len(city_dir)):
        frame_dir = image_dir + city_dir[i]
        back_dir = inpainted_back_dir + city_dir[i]
        frame_depth_dir = depth_dir + city_dir[i]
        
        frame_list = os.listdir(frame_dir)
        frame_list.sort()
        for j in range(len(frame_list)//30):
            image = []
            back = []
            instance = []
            semantic = []
            instance_npy = []
            depth = []
            for k in range(j*30 + 0, (j+1)*30):
                full_image_path = frame_dir + "/" + frame_list[k]
                assert os.path.isfile(full_image_path)
                full_back_path = back_dir + "/" + frame_list[k]
                #if k != (j*30 + 19) or (phase is 'test'):
                full_instance_path = instance_mask_rcnn_dir + city_dir[i] + "/" + frame_list[k]
                full_semantic_path = semantic_psp_dir + city_dir[i] + "/" +  frame_list[k]
                full_depth_path = frame_depth_dir + "/" +  frame_list[k][:-3] + "npy"
                image.append(full_image_path)
                back.append(full_back_path)
                instance.append(full_instance_path)
                semantic.append(full_semantic_path)
                depth.append(full_depth_path)
            track_instance_video_dir = track_instance_dir + "%04d/"%cnt
            cnt = cnt + 1
            track_dict = load_tracked_dict_val(track_instance_video_dir)
            video.append((image, back, instance, semantic, track_dict, depth))

    return video

# Just save the dict of all paths for saving loading data time

image_dir = '/root/data/cityscapes/leftImg8bit_sequence_512p/train/'
track_instance_dir = '/root/data/cityscapes/fore_train_data/train/'
car_mask_dir = '/root/data/cityscapes/car_mask/train/'
semantic_dir = '/root/data/cityscapes/semantic_new/train/'
inpainted_back_dir = '/root/data/cityscapes/leftImg8bit_sequence_512p_background_inpainted/train/'
city_dir = os.listdir(image_dir)
city_dir.sort()
cnt = 0
video = []



for i in range(len(city_dir)):
    frame_rgb_dir = image_dir + city_dir[i]
    frame_car_dir = car_mask_dir + city_dir[i]
    frame_semantic_dir = semantic_dir + city_dir[i]
    frame_back_dir = inpainted_back_dir + city_dir[i]
    frame_list = os.listdir(frame_rgb_dir)
    frame_list.sort()
    for j in range(len(frame_list)//30):
        print("Processing city directory %d frame %d"%(i, j))
        image = []
        car_mask = []
        semantic = []
        back = []
        for k in range(j*30 + 0, (j+1)*30):
            full_image_path = frame_rgb_dir + "/" + frame_list[k]
            full_car_path = frame_car_dir + "/" +  frame_list[k]
            full_semantic_path = frame_semantic_dir + '/' + frame_list[k]
            full_back_path = frame_back_dir + '/' + frame_list[k]
            image.append(full_image_path)
            back.append(full_back_path)
            car_mask.append(full_car_path)
            semantic.append(full_semantic_path)
        track_instance_video_dir = track_instance_dir + "%04d/"%cnt
        video = load_tracked_dict(track_instance_video_dir, image, car_mask, semantic, back, video)
        cnt = cnt + 1
    np.save("train_list.npy", video)