import numpy as np
import scipy
import os
from PIL import Image
import glob
import pandas as pd
import cv2





def read_image(path, h=None, w=None):
    return np.array(Image.open(path))

def IOU_mask(moving_mask, obj):
    moving_mask = moving_mask
    obj = obj
    return (moving_mask * obj).sum() / obj.sum()



origin_class = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
map_class    = [0,1,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18]




def class_mapping(cls):
    mapping = origin_class.index(cls)
    return mapping


def loadinclass(class_file_path):
    f = open(class_file_path, 'r')
    data = f.readlines()
    num_objs = len(data)
    class_arr = []
    for i in range(num_objs):
        cnt_data = data[i]
        split = cnt_data.split(" ")
        name = split[0].split("/")[-1]
        cls = int(split[1])
        mapping = class_mapping(cls)
        class_arr.append((name, mapping))
    return class_arr


def get_mask(frames):
    if len(frames) == 1:
        rider_mask = frames[0]
    elif len(frames) == 4:
        rider_mask = frames[3]
    return rider_mask

def interaction_mask(rider_mask, bike_mask):
    kernel = np.ones((5,5),np.uint8)
    rider_mask = cv2.dilate(rider_mask.astype(np.float32), kernel, iterations=2)
    bike_mask = cv2.dilate(bike_mask.astype(np.float32), kernel, iterations=2)
    rider_mask = rider_mask.astype(np.bool)
    bike_mask = bike_mask.astype(np.bool)
    return 1.0 * (rider_mask & bike_mask).astype(np.int32).sum() / rider_mask.astype(np.int32).sum()

def combine_bike_rider(instance_list, rider_id, bike_id):
    #rider_frames
    rider_frames = instance_list[rider_id][0]
    rider_num = len(rider_frames)
    #bike_frames
    bike_frames = instance_list[bike_id][0]
    bike_num = len(bike_frames)
    if rider_num == 1 or bike_num == 1:
        # If any of rider or bike only have one frame
        # Combine the last frame and wait for bwd warp to generate correct mask
        rider_last_frame = get_mask(rider_frames)
        bike_last_frame = get_mask(bike_frames)
        return [(rider_last_frame | bike_last_frame).astype(np.int32)]
    else:
        # Both of them have 4 frames
        # Combine frame one by one
        combine_frames = []
        for k in range(4):
            rider_cnt_frame = rider_frames[k]
            bike_cnt_frame = bike_frames[k]
            combine_cnt_frame = (rider_cnt_frame | bike_cnt_frame).astype(np.int32)
            combine_frames.append(combine_cnt_frame)
        return combine_frames


def preprocess_bike_person(instance_list):
    valid_index = np.zeros(len(instance_list)) + 1
    #1 represent valid, 0 represent have been used
    # classes
    # 'rider'-12 moto-17 bike-18 
    # valid_class = [12, 17, 18]
    new_instance_list = []
    for i in range(len(instance_list)):
        if valid_index[i] == 0:
            continue
        curr_data = instance_list[i]
        if curr_data[1] != 12:
            continue
        frames = curr_data[0]
        # Compute rider mask
        rider_mask = get_mask(frames)
        # Search among others to find match bike
        iou_score = -1
        bike_id = -1
        for k in range(len(instance_list)):
            cnt_data = instance_list[k]
            if cnt_data[1] in [17, 18]:
                # Check whether match
                cnt_mask = get_mask(cnt_data[0])
                iou = interaction_mask(rider_mask, cnt_mask)
                if iou > iou_score:
                    iou_score = iou
                    bike_id = k
        if iou_score > 0:
            # Find corrsponding bike
            combined_segs = combine_bike_rider(instance_list, i, bike_id)
            new_instance_list.append((combined_segs, 12))
            valid_index[i] = 0
            valid_index[bike_id] = 0
    for k in range(len(instance_list)):
        if valid_index[k] == 1:
            new_instance_list.append((instance_list[k][0], instance_list[k][1]))
    return new_instance_list

def process_dynamic(object_list, dynamic_path):
    move_mask = 1.0 - np.array(Image.open(dynamic_path[3]).resize((1242, 375)))/255.0
    new_list = []
    for i in range(len(object_list)):
        #print(object_list[i])
        #print(len(object_list[i][0]))
        if len(object_list[i][0]) == 4:
            cnt_obj = object_list[i][0][3]
        else:
            #print(object_list[i][0].shape)
            cnt_obj = object_list[i][0][0]
        #print(cnt_obj.shape)
        #Image.fromarray((cnt_obj*255).astype(np.uint8)).save("./debug/%02d.png"%i)
        #print("iou", IOU_mask(move_mask, cnt_obj) > 0.2)
        #print("move mask shape", move_mask.shape)
        #print("cnt obj shape", cnt_obj.shape)
        if IOU_mask(move_mask, cnt_obj) > 0.4:
            new_list.append((object_list[i][0], object_list[i][1]))
    return new_list



def load_tracked_dict(path, images, semantics, backs, depths, dynamics, class_file, segs_path, cnt_id):
    classes = loadinclass(class_file)
    num_objs = len(classes) 
    #print("Class number from origin instance data =", num_objs)
    #Load all masks and classes into a dict
    all_object_dict = []
    for i in range(num_objs):
        cnt_object_dir = path + "%02d/"%i
        #print("have tracking result")
        frames = []
        if os.path.exists(cnt_object_dir):
            frames_path = glob.glob(cnt_object_dir + "*.png")
            frames_path.sort()
            for j in range(len(frames_path)):
                cnt = np.array(Image.open(frames_path[j]))/255
                frames.append(cnt.astype(np.int32))
        else:
            frames_path = segs_path + classes[i][0]
            cnt = np.array(Image.open(frames_path).resize((1242, 375), Image.NEAREST))/255
            frames.append(cnt.astype(np.int32))
        all_object_dict.append((frames, classes[i][1]))
        #print("object %02d len %02d class %d"%(i,len(frames), classes[i][1]))
    ### Combine rider with corrsponding bike
    #print("all objects", len(all_object_dict))
    processed_list = preprocess_bike_person(all_object_dict)
    #print("process bike", len(processed_list))
    processed_list = process_dynamic(processed_list, dynamics)
    #print("process dynamic", len(processed_list))
    #print("Processed list len = ", len(processed_list))
    if len(processed_list) > 0:
        return [images, semantics, backs, depths, dynamics, processed_list, cnt_id]
    else:
        return []



# Just save the dict of all paths for saving loading data time
# Script for saving 
image_dir = './data/kitti/raw_data_256p/val/'

# tracking result
track_instance_dir = './data/kitti/fore_val_data/val/'
semantic_dir = './data/kitti/semantic/val/gray/'
depth_dir = './data/depth/val/'

my_back_dir = "./result/kitti/"
instance_origin_txt_dir = './data/kitti/instance_upsnet/origin_data/val/txt/'
instance_origin_segs_dir = './data/kitti/instance_upsnet/origin_data/val/segs/'
city_dir = os.listdir(image_dir)
city_dir.sort() 
cnt = 0
video = []

for i in range(len(city_dir)):
    frame_rgb_dir = image_dir + city_dir[i] + "/image_02/data/"
    frame_semantic_dir = semantic_dir + city_dir[i] + "/image_02/data_transformed/"
    frame_depth_dir = depth_dir + city_dir[i] + "/image_02/data/"
    frame_instance_segs_dir = instance_origin_segs_dir + city_dir[i] + "/"
    frame_instance_txt_dir = instance_origin_txt_dir + city_dir[i] + "/"
    frame_list = os.listdir(frame_rgb_dir)
    frame_list.sort()

    track_instance_cnt_dir = track_instance_dir + "%04d/"%i
    for j in range(len(frame_list)- 8):
        print("processing city %02d frame %04d"%(i,j))
        image = []
        depth = []
        semantic = []
        # Load all inpainted background to test whether foreground prediction model worked
        # Load input 4 frames background for input data
        for k in range(j, j+ 9):
            full_image_path = frame_rgb_dir + "/" + frame_list[k]
            #print(full_image_path)
            assert os.path.isfile(full_image_path)
            image.append(full_image_path)

            full_depth_path = frame_depth_dir + "/" + frame_list[k][:-3]+"npy"
            assert os.path.isfile(full_depth_path)
            depth.append(full_depth_path)
            
            full_semantic_path = frame_semantic_dir + '/' + frame_list[k]
            assert os.path.isfile(full_semantic_path)
            semantic.append(full_semantic_path)
        dynamic_mask = []
        back = []
        cnt_my_back = my_back_dir + "%04d/"%cnt
        for k in range(4):
            full_back_path = cnt_my_back + "input_image_inpainted_%02d.png"%k
            #assert os.path.isfile(full_back_path)
            back.append(full_back_path)
        for k in range(6):
            full_back_path = cnt_my_back + "warp_image_bwd_inpainted_%02d.png"%k
            #assert os.path.isfile(full_back_path)
            back.append(full_back_path)
        for k in range(4):
            dynamic_mask_path = cnt_my_back + "input_mask_%02d.png"%k
            #assert os.path.isfile(dynamic_mask_path)
            dynamic_mask.append(dynamic_mask_path)

        segs_path = frame_instance_segs_dir + frame_list[j+3][:-4] + "/"
        assert os.path.exists(segs_path)
        class_file = frame_instance_txt_dir + frame_list[j+3][:-4] + "pred.txt"
        assert os.path.isfile(class_file)
        track_instance_video_dir = track_instance_cnt_dir + "/%02d/"%(j+3)
        assert os.path.exists(track_instance_video_dir)
        video = load_tracked_dict(track_instance_video_dir, image, semantic, back, depth, dynamic_mask, class_file, segs_path, cnt)
        if len(video) > 0:
            os.makedirs("./test_kitti/", exist_ok=True)
            np.save("./test_kitti/test_kitti_my_back_%05d.npy"%(cnt), video)
        cnt = cnt + 1
        print("processing cnt", cnt)
