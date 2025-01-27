from collections import Counter
import cv2
import math
import os
from PIL import Image, ImageOps
import shutil
import json
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import argparse
import numpy as np
import os
import struct
from tqdm import tqdm
def mk_meta_json(dir_path,subset=[]):
    # For Creating YT-VOS format for training data
    # Initialize the JSON data structure
    data = {
        "videos": {}
    }
    # Loop through each directory in the directory path or not, if subset isn't empty.
    loopie = os.listdir(dir_path) if subset == [] else [str(x) for x in subset]

    for item in loopie:
        if os.path.isdir(os.path.join(dir_path, item)):
            # Initialize the object data structure
            object_data = {
                "objects": {
                    "1": {
                        "category": None,
                        "frames": []
                    }
                }
            }
            
            # Loop through each file in the directory and add the frame names to the object data structure
            for file_name in sorted(os.listdir(os.path.join(dir_path, item))):
                if file_name.endswith('.jpg'):
                    object_data['objects']['1']['frames'].append(file_name.split('.')[0])
            
            # Add the object data structure to the JSON data structure
            data["videos"][item] = object_data
    return data


def mov_to_frames(mov_file_path, img_save_file_path, wish_frames):
    # Open the video file
    video = cv2.VideoCapture(mov_file_path)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")

    # fps
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    save_int = int(round(total_frames/wish_frames, 0))

    if not os.path.exists(img_save_file_path.split('.MOV')[0]):
        os.makedirs(img_save_file_path.split('.MOV')[0])

    # Loop through the video frames
    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # Check if we have reached the end of the video
        if not ret:
            break
        
        #Send first few images to be annotated using SAM:
        # # in case SAM doesn't spot in first image, annotate all obj in first 3 images using SAM
        # if frame_count < 3:
        #     cv2.imwrite(fr"{unlabeled_annot_save_path.split('.MOV')[0]}/{str(frame_count).zfill(4)}.jpg", frame)

        # Save the current frame as a JPEG image
        if frame_count % save_int == 0:
            cv2.imwrite(fr"{img_save_file_path.split('.MOV')[0]}/{str(frame_count).zfill(4)}.jpg", frame)
        
        #next frame
        frame_count += 1

    # Release the video file
    video.release()

def resize_images_in_folder(folder_path, target_width, target_height):
    """
    Resize all images in the specified folder to the target dimensions
    and overwrite the original images.
    
    Args:
        folder_path (str): Path to the folder containing images.
        target_width (int): Target width for resizing.
        target_height (int): Target height for resizing.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping invalid image: {file_path}")
            continue
        
        # Resize the image
        resized_image = cv2.resize(image, (target_width, target_height))
        
        # Overwrite the original image
        cv2.imwrite(file_path, resized_image)
        print(f"Resized and saved: {file_path}")


def VOS_format_data(input_dataset_path, obj_name,vos_dataset_path,height=1080,width=1920,times_255=False,resiz=True):
    
    for i in os.listdir(rf'{input_dataset_path}/JPEGImages/{obj_name}'): #range(len(os.listdir())): #
        image = Image.open(fr'{input_dataset_path}/JPEGImages/{obj_name}/{i}')    
        if resiz:
            new_image = image.resize((width,height))
        else: 
            new_image = image
        #make save folder if not exist
        if not os.path.exists(fr'{vos_dataset_path}/JPEGImages/{obj_name}'):
            os.makedirs(fr'{vos_dataset_path}/JPEGImages/{obj_name}')
        new_image.save(fr'{vos_dataset_path}/JPEGImages/{obj_name}/{i}')

    #make save folder if not exist
    if not os.path.exists(fr'{vos_dataset_path}/Annotations/{obj_name}'):
        os.makedirs(fr'{vos_dataset_path}/Annotations/{obj_name}')

    annotation_name = os.listdir(fr'{input_dataset_path}/Annotations/{obj_name}')
    if os.path.exists(fr'{input_dataset_path}/Annotations/{obj_name}/{annotation_name[0]}'):
        if len(annotation_name) != 1:
            print(f'More than one annotations in object: {obj_name} at path: {input_dataset_path}/Annotations/')
            sys.exit()
        else: 
            image = Image.open(fr'{input_dataset_path}/Annotations/{obj_name}/{annotation_name[0]}')
            if resiz:
                new_image = image.resize((width,height))
            else: 
                new_image = image
            new_image = ImageOps.grayscale(new_image)
            pixels = new_image.load() # create the pixel map
        
            if times_255:
                val=255
            else:
                val=1
            for i in range(new_image.size[0]): # for every pixel:
                for j in range(new_image.size[1]):
                    if pixels[i,j] != 0:
                        # change to black if not red
                        pixels[i,j] = val
            
            if not os.path.exists(fr'{vos_dataset_path}/Annotations/{obj_name}'):
                os.makedirs(fr'{vos_dataset_path}/Annotations/{obj_name}')
            new_image.save(fr'{vos_dataset_path}/Annotations/{obj_name}/{annotation_name[0]}')

import json
import numpy as np


def mk_test_image_folder(transforms_json_path,test_out_path):
    with open(transforms_json_path) as f:
        test_data = json.load(f)
    
    if os.path.exists(test_out_path):
        print('test images already exist at test_out_path:',test_out_path)
        return
    else:
        os.mkdir(test_out_path)

    for x in test_data['frames']:
        img_path = x["file_path"]
        img_id = img_path.rsplit('/',1)[-1]
        new_img_path = os.path.join(test_out_path,img_id)
        shutil.copy(img_path,new_img_path)
    print('successfully copied to: ', test_out_path)
    return 

def mk_nerf_loss_json(transforms_json_path,root_dir,train_test_split=0.8, mip_style=True,reproj_folder=True,obj=''):

    if mip_style:
        with open(transforms_json_path) as f:
            train_data = json.load(f)
        with open(transforms_json_path) as f:
            test_data = json.load(f)

        lffhold=8

        num_imgs = len(train_data['frames'])
        all_index = [x for x in range(0, num_imgs)]
        test_index = [x for x in range(0, num_imgs, lffhold)]

        print('All images, N:',len(all_index))
        print('Test images (every 8th img):',len(test_index))


        train_data['frames'] = sorted(train_data['frames'].copy(), key = lambda x : int(x["file_path"].rsplit('/',1)[-1][:-4]))
        test_data['frames'] = sorted(test_data['frames'].copy(), key = lambda x : int(x["file_path"].rsplit('/',1)[-1][:-4]))


        train_data['frames'] = [train_data['frames'][x] for x in all_index if (x not in test_index)]
        test_data['frames'] = [test_data['frames'][x] for x in all_index if (x in test_index)]

        # train_data['frames'].insert(0, test_data['frames'][0])
        # test_data['frames'] = test_data['frames'][1:] #optimally, first frame should be part of train set also bcs we choose 1. mask using SAM but...
        if not os.path.exists(f"data/undistorted_images/test_images_uncorrected/masked_JPEGImages/{obj}"):
            os.makedirs(f"data/undistorted_images/test_images_uncorrected/masked_JPEGImages/{obj}")
            for fram in test_data['frames']:
                shutil.copy(f"data/undistorted_images/XMEM/JPEGImages/{obj}/" + fram['file_path'].rsplit('/',1)[-1], f"data/undistorted_images/test_images_uncorrected/masked_JPEGImages/{obj}/" + fram['file_path'].rsplit('/',1)[-1])
        if not os.path.exists(f"data/undistorted_images/test_images_uncorrected/masks/{obj}"):
            os.makedirs(f"data/undistorted_images/test_images_uncorrected/masks/{obj}")
            for fram in test_data['frames']:
                shutil.copy(f"data/undistorted_images/XMEM/Annotations/{obj}/" + fram['file_path'].rsplit('/',1)[-1].replace('.jpg','.png'), f"data/undistorted_images/test_images_uncorrected/masks/{obj}/" + fram['file_path'].rsplit('/',1)[-1].replace('.jpg','.png'))        

        if reproj_folder:
            seg_type="Please name your reprojection folder with the string MVG..." 
            for fram in train_data['frames']:

                subelements = fram['file_path'].split('/')
                
                for subelement in subelements:
                    if "MVG" in subelement:
                        seg_type = subelement
                fram['file_path'] = f"{root_dir}/data/undistorted_images/{seg_type}/{obj}/"+ fram['file_path'].rsplit('/',1)[-1]
            
            seg_type="Please name your reprojection folder with the string MVG..." 
            for fram in test_data['frames']:

                subelements = fram['file_path'].split('/')
                
                for subelement in subelements:
                    if "MVG" in subelement:
                        seg_type = subelement
                fram['file_path'] = f"{root_dir}/data/undistorted_images/{seg_type}/{obj}/" + fram['file_path'].rsplit('/',1)[-1]


        output_dir = transforms_json_path.rsplit('/',1)[0]

        with open(fr'{output_dir}/transforms_train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(fr'{output_dir}/transforms_test.json', 'w') as f:
            json.dump(test_data, f, indent=2)
    


    return 

def create_gif_from_images(directory, gif_filename, frame_duration):
    """
    Creates a GIF file from a series of images in a directory.
    
    Args:
    - directory (str): the directory where the images are stored
    - gif_filename (str): the name of the output GIF file
    - frame_duration (int): the duration of each frame in milliseconds
    
    Returns:
    - None
    """
    
    if not os.path.exists(os.path.dirname(gif_filename)):
        os.makedirs(os.path.dirname(gif_filename))

    # get a list of all the image filenames in the directory
    image_filenames = [f for f in os.listdir(directory) if f.endswith('.png')]

    if not image_filenames:
        print('no png images found, trying to fin jpeg instead')
        image_filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        if image_filenames:
            print('found jpg images')

    # sort the image filenames by their numerical order
    image_filenames = sorted(image_filenames, key=lambda x: int(x.split('.')[0]))

    # create a list of PIL image objects from the image filenames
    images = [Image.open(os.path.join(directory, f)) for f in image_filenames]

    # save the list of images as a GIF file with the given frame duration
    images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=frame_duration, loop=0)



def create_gif_from_images_use_for_render(directory, gif_filename, frame_duration):
    """
    Creates a GIF file from a series of images in a directory.
    
    Args:
    - directory (str): the directory where the images are stored
    - gif_filename (str): the name of the output GIF file
    - frame_duration (int): the duration of each frame in milliseconds
    
    Returns:
    - None
    """
    
    if not os.path.exists(os.path.dirname(gif_filename)):
        os.makedirs(os.path.dirname(gif_filename))

    # get a list of all the image filenames in the directory
    image_filenames = [f for f in os.listdir(directory) if f.endswith('.png')]

    if not image_filenames:
        print('no png images found, trying to fin jpeg instead')
        image_filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        image_filenames = [f for f in image_filenames if not f.startswith('masked_')]

        if image_filenames:
            print('found jpg images')
    else:
        image_filenames = [f for f in image_filenames if not f.startswith('masked_')]


    # sort the image filenames by their numerical order
    image_filenames = sorted(image_filenames, key=lambda x: int(x.split('.')[0]))

    # create a list of PIL image objects from the image filenames
    images = [Image.open(os.path.join(directory, f)) for f in image_filenames]

    # save the list of images as a GIF file with the given frame duration
    images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=frame_duration, loop=0)


# For reading depth maps.
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


### Functions for Choosing sam masks---------------------
def imgBitwise(dir_img_src, dir_img_target):
    img_src = cv2.imread(dir_img_src,0)
    img_target = cv2.imread(dir_img_target,0)
    
    img_src_1s = np.sum(img_src == 255)
    img_target_1s = np.sum(img_target == 255)
    
    bitwise_1s = np.bitwise_and(img_src,img_target)
    bitwise_1s_sum = np.sum(bitwise_1s == 255)
    
    return bitwise_1s_sum, img_src_1s, img_target_1s
def mask_overlap(mask1, mask2,change_in_size_weight=1):

    # Compute the intersection
    intersection = cv2.bitwise_and(mask1, mask2)
    total_pixels = cv2.bitwise_or(mask1,mask2)

    # Count the number of intersection pixels
    intersection_pixels = cv2.countNonZero(intersection)
    aglomeration_pixels = cv2.countNonZero(total_pixels)
    # Calculate the overlap percentage
    #total_pixels = mask1.size
    mask1_1s = np.sum(mask1 == 255)
    mask2_1s = np.sum(mask2 == 255)
    #total_pixels = mask1_1s + mask2_1s
    overlap = (intersection_pixels / aglomeration_pixels)*(1 - abs(change_in_size_weight*((mask1_1s - mask2_1s) / (mask1_1s + mask2_1s))))

    #print(f"Overlap percentage: {overlap_percentage:.2f}%")
    return overlap

def choose_mask(dir_mask_path, img_path, output_dir, first_img, first_mask_path,col,change_in_size_weight,conf_thresh=0.5):
    for image_folder in tqdm(sorted(os.listdir(dir_mask_path))):
        if first_img == True:
            first_img = False        
            best_mask_path = first_mask_path
            first_img_path = os.path.join(img_path,"0000.jpg")
            first_img_full = cv2.imread(first_img_path)
            best_mask = cv2.imread(best_mask_path,cv2.IMREAD_GRAYSCALE)
            best_seg_img = cv2.bitwise_and(first_img_full, first_img_full, mask=best_mask)
            best_tot_score = 0.5
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB) if no plot rgb dont include line        
            if col:
                best_hist_b = cv2.calcHist([best_seg_img],[0],None,[256],[1,256])
                best_hist_g = cv2.calcHist([best_seg_img],[1],None,[256],[1,256])
                best_hist_r = cv2.calcHist([best_seg_img],[2],None,[256],[1,256])
            cv2.imwrite(os.path.join(output_dir,f'{image_folder}.png'), best_mask)
            continue
        #best is best in last folder, while cur is current best while exploring - it becomes the best after exploration unless very low confidence. 

        image_full = cv2.imread(os.path.join(img_path,f'{image_folder}.jpg'))

        masks_for_image_path = os.path.join(dir_mask_path,image_folder)
        #iterate over all masks and keep track of current best score.
        cur_tot_score = 0
        for mask_item in sorted(os.listdir(masks_for_image_path)):
            if mask_item.endswith(".png"):
                mask_path = os.path.join(masks_for_image_path,mask_item)
                mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
                seg_img = cv2.bitwise_and(image_full, image_full, mask=mask)

                overlap_score = mask_overlap(best_mask, mask,change_in_size_weight)
                if col:
                    hist_b = cv2.calcHist([seg_img],[0],None,[256],[1,256])
                    hist_g = cv2.calcHist([seg_img],[1],None,[256],[1,256])
                    hist_r = cv2.calcHist([seg_img],[2],None,[256],[1,256])

                    b_score = float(cv2.compareHist(best_hist_b, hist_b, cv2.HISTCMP_CORREL))
                    g_score = float(cv2.compareHist(best_hist_g, hist_g, cv2.HISTCMP_CORREL))
                    r_score = float(cv2.compareHist(best_hist_r, hist_r, cv2.HISTCMP_CORREL))
                    bgr_score = (b_score + g_score + r_score)/3
                    tot_score = overlap_score*(1-col) + bgr_score*(col)
                else:
                    tot_score = overlap_score

                if tot_score > cur_tot_score:
                    cur_tot_score = tot_score

                    cur_mask = mask
                    cur_mask_path = mask_path
                    if col:
                        cur_hist_b = hist_b
                        cur_hist_g = hist_g
                        cur_hist_r = hist_r
                # if str(image_folder) == "0296": #and (str(mask_item) == "0.png" or str(mask_item) == "6.png"):
                #     print(str(mask_item), overlap_score,'\n', bgr_score,'\n', tot_score)        
        if (best_tot_score - cur_tot_score) < conf_thresh:
            best_mask = cur_mask
            best_mask_path = cur_mask_path
            best_tot_score = cur_tot_score
            if col:
                best_hist_b = cur_hist_b
                best_hist_g = cur_hist_g
                best_hist_r = cur_hist_r
        else:
            print(f"skipped at mask_item: {cur_mask_path} with tot score:  {cur_tot_score}")
        
        contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 300]

        filtered_mask = np.zeros_like(cur_mask)
        cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

        cur_mask = filtered_mask
        cv2.imwrite(os.path.join(output_dir,f'{image_folder}.png'), filtered_mask)

def combine_masks(masks, ref_mask, ref_mask_old, img_path, image_folder, obj_name, col=0.0, conf_threshold=0.5):
    #best one of them, I think.
    ref_img = cv2.imread(img_path)    
    if col:
        best_hist_b = cv2.calcHist([ref_img],[0],None,[256],[1,256])
        best_hist_g = cv2.calcHist([ref_img],[1],None,[256],[1,256])
        best_hist_r = cv2.calcHist([ref_img],[2],None,[256],[1,256])

    selected_masks = []
    reference_mask = np.logical_or(ref_mask, ref_mask_old)
    new_reference_mask = np.zeros_like(reference_mask)
    for mask in masks:
        intersection = np.logical_and(mask, reference_mask)
        if intersection.any():
            intersection_mask = intersection.astype('uint8')*255
            overlap_score = mask_overlap(mask,intersection_mask)
            if col:
                intersection_img = cv2.bitwise_and(ref_img,ref_img,mask=intersection_mask)

                intersection_mask_hist_b = cv2.calcHist([intersection_img],[0],None,[256],[1,256])
                intersection_mask_hist_g = cv2.calcHist([intersection_img],[1],None,[256],[1,256])
                intersection_mask_hist_r = cv2.calcHist([intersection_img],[2],None,[256],[1,256])
            
                b_score = float(cv2.compareHist(best_hist_b, intersection_mask_hist_b, cv2.HISTCMP_CORREL))
                g_score = float(cv2.compareHist(best_hist_g, intersection_mask_hist_b, cv2.HISTCMP_CORREL))
                r_score = float(cv2.compareHist(best_hist_r, intersection_mask_hist_b, cv2.HISTCMP_CORREL))
                bgr_score = (b_score + g_score + r_score)/3
                tot_score = overlap_score*(1-col) + bgr_score*(col)
            else:
                tot_score = overlap_score
            if (tot_score >= conf_threshold): #and mask_pixels < 10000) or intersection_score >= 0.5:
                selected_masks.append(mask)
                #print('trigger')
                new_reference_mask = np.logical_or(new_reference_mask, mask)

    new_reference_mask = new_reference_mask.astype(np.uint8) * 255

    new_img = cv2.bitwise_and(ref_img,ref_img,mask=new_reference_mask)

    cv2.imwrite(f'/work3/s204161/BachelorData/bachelor_data/masks/SAM_collect_masks/Annotations/{obj_name}/{image_folder}.png',ref_mask)
    cv2.imwrite(f'/work3/s204161/BachelorData/bachelor_data/masks/SAM_collect_masks/JPEGImages/{obj_name}/{image_folder}.jpg',new_img)

    return new_reference_mask,ref_mask



#-----------------------------------------------


#Functions for forward projection
def load_point_cloud(raw_volume_path,siz,channels=4,offset=0.5,scale=0.33,visualize=False,threshold_density=200):
    with open(raw_volume_path, mode='rb') as f:
        XYZ_1 = np.fromfile(f, dtype='float32') #
        mesh_1 = XYZ_1.reshape(siz,siz,siz,channels)

    mesh_b = mesh_1[:,:,:,3] #object is blue, so using blue channel is fine, (3-channel) density also works...

    mesh_mask = mesh_b > threshold_density #300 #350.0 #0.08
    mesh_scale=1
    grid_frequency, radius, offset = siz, 0.5, 0.5
    list3d = []

    lim1, lim2 = -radius + offset, radius + offset #+ offset, radius # + offset

    for x in np.linspace(lim1,lim2, grid_frequency):
        x_int = int(((x - lim1) * grid_frequency) -1)
        for y in np.linspace(lim1,lim2, grid_frequency):
            y_int = int((y - lim1) * grid_frequency -1)
            for z in np.linspace(lim1,lim2, grid_frequency):
                z_int = int((z - lim1) * grid_frequency -1)
                if mesh_mask[x_int,y_int,z_int]:
                    list3d.append([z*mesh_scale, x*mesh_scale, y*mesh_scale]) # (result.y, result.z, result.x) 

    print('num_points:',len(list3d))

    list3d = [ngp_position_to_nerf(point, scale, offset) for point in list3d]

    if visualize:
        clear_output()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_coords = [point[0] for point in list3d]
        y_coords = [point[1] for point in list3d]
        z_coords = [point[2] for point in list3d]

        ax.scatter(x_coords, y_coords, z_coords, color=(0, 0, 1), label=f'Class {0+1}', marker="s")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        rad, off=1,0.
        ax.set_xlim(-rad+off,rad+off)
        ax.set_ylim(-rad+off,rad+off)
        ax.set_zlim(-rad+off,rad+off)

        display(fig)
        ax.view_init(elev=0, azim=90)
    return np.asarray(list3d)

    



def ngp_position_to_nerf(pos, scale, offset):
    new_pos = np.copy(pos)
    
    new_pos[0] = pos[2]
    new_pos[1] = pos[0]
    new_pos[2] = pos[1]

    new_pos = (new_pos - offset) / scale
    #new_pos[[0,1,2]] = new_pos[1,0,2]
    return new_pos

from collections import defaultdict
# import numpy as np
# import os
# from tqdm import tqdm


def project_points(transforms_path, list3d, path_to_masks="", save_points=""):
    """
    function for projecting points.

    transforms_path is path to transforms.json with camera poses in nerf coordinate system
    list3d contains the 3d point cloud which are input candidates for the reprojection
    path_to_masks is path to the masks for which we constrain the reprojection to be inside.

    returns a list containing all 3d points that were inside an image for each images when doing forward projection.
    """

    with open(transforms_path) as f:
        transforms_json = json.load(f)

    k1 = transforms_json['k1']
    k2 = transforms_json['k2']
    k3 = transforms_json['k3']
    k4 = transforms_json['k4']
    p1 = transforms_json['p1']
    p2 = transforms_json['p2']
    C_x = transforms_json['cx']  # int(img1.shape[0]/2)
    C_y = transforms_json['cy']  # int(img1.shape[1]/2)
    f_x = transforms_json['fl_x']
    f_y = transforms_json['fl_y']
    h = int(transforms_json['h'])
    w = int(transforms_json['w'])
    #sort frames by order, 0000 first...
    transforms_json['frames'] = sorted(transforms_json['frames'],
                                            key=lambda x: int(x["file_path"].split("/")[-1].split(".")[0]))
    comp_list3d = []

    dist_coefs = None #np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)

    int_matrix = np.array([[-f_x, 0, C_x],
                    [0, f_y, C_y],
                    [0, 0, 1]])

    #print("Intrinsic camera matrix: \n", int_matrix)
    tqdm_bar = tqdm(transforms_json['frames']) #tqdm(colmap_transforms_json['frames'])
    #results = np.zeros((len(transforms_json['frames']), h, w),dtype='uint8')

    results = np.zeros((len(transforms_json['frames']), h,w),dtype='uint8')

    c=0
    for im_data in tqdm_bar:
        transf_matrix = np.array(im_data['transform_matrix'])

        transf_matrix_inv = np.linalg.inv(transf_matrix)

        R = transf_matrix_inv[:3,:3]
        t = transf_matrix_inv[:3,3]

        rvec, _ = cv2.Rodrigues(R)
        cam_pos = t
    
        list3d_hit = []
        
        if path_to_masks:
            path_mask = path_to_masks + im_data['file_path'].split('/')[-1].replace('.jpg', '.png') 
            img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) # 
            #img_mask = cv2.undistort(img_mask,int_matrix,dist_coefs)
            h, w = img_mask.shape
            for p in list3d:                
                p = np.expand_dims(p, axis=-1)
                uv, _ = cv2.projectPoints(p, rvec, t, int_matrix, dist_coefs)
                uv = uv/1.
                u, v = int(round(uv[0,0,0])), int(round(uv[0,0,1]))
                
                #if inside img, then if inside mask:
                if 0 <= u and u < w and 0 <= v and v < h:
                    # we index using v,u, since img[index what row / y, index what column / x]
                    #result[v,u] = 1
                    if img_mask[v,u]:
                        list3d_hit.append([cord[0] for cord in p])

            # p = np.expand_dims(list3d, axis=-1)
            # uv, _ = cv2.projectPoints(p, R, t, int_matrix, dist_coefs)
            # uv = uv / 1.
            # u = np.round(uv[..., 0]).astype(int)
            # v = np.round(uv[..., 1]).astype(int)
            
            # # if w > h:
            # valid_indices = np.logical_and.reduce([u >= 0, u < w, v >= 0, v < h])
            # # else:
            # #     valid_indices = np.logical_and.reduce([u >= 0, u < h, v >= 0, v < w])
            # u_valid = u[valid_indices]
            # v_valid = v[valid_indices]

            # inside_mask = img_mask[v_valid, u_valid]

            # inside_indices = np.where(inside_mask)

            # if len(inside_indices[0]) == 1:
            #     list3d_hit = list3d[inside_indices[0][0]]
            #     comp_list3d.append([list3d_hit])
            #     continue

            # list3d_hit = [list3d[i] for i in np.array(inside_indices).squeeze()]

            comp_list3d.append(list3d_hit)
        else:
            
            result = np.zeros((h,w))
            p = np.expand_dims(list3d, axis=-1)
            uv, _ = cv2.projectPoints(p, rvec, t, int_matrix, dist_coefs)
            uv = uv / 1.
            u = np.round(uv[..., 0]).astype(int)
            v = np.round(uv[..., 1]).astype(int)
    
            valid_indices = np.logical_and.reduce([u >= 0, u < w, v >= 0, v < h])

            u_valid = u[valid_indices]
            v_valid = v[valid_indices]

            result[v_valid, u_valid] = 1
            results[c,:,:] = result
            c+=1

            # result = np.zeros((h,w))
            # p = np.expand_dims(list3d, axis=-1)
            # uv, _ = cv2.projectPoints(p, rvec, t, int_matrix, dist_coefs)
            # # uv = uv / 1.
            # # u = np.round(uv[..., 0]).astype(int)
            # # v = np.round(uv[..., 1]).astype(int)

            # for p in uv:
            #     p = p/1

            #     u, v = int(round(p[0,0])), int(round(p[0,1]))
            #     if 0 <= u and u < w and 0 <= v and v < h:
            #         # we index using v,u, since img[index what row / y, index what column / x]
            #         result[v,u] = 1    
            
            # results += [result]

            # for p in list3d:                
            #     p = np.expand_dims(p, axis=-1)
            #     uv, _ = cv2.projectPoints(p, rvec, t, int_matrix, dist_coefs)
            #     uv = uv/1.
            #     u, v = int(round(uv[0,0,0])), int(round(uv[0,0,1]))
                
            #     #if inside img, then if inside mask:
            #     if 0 <= u and u < w and 0 <= v and v < h:
            #         # we index using v,u, since img[index what row / y, index what column / x]
            #         result[v,u] = 1
            # results += [result]
            # uv, _ = cv2.projectPoints(p, rvec, t, int_matrix, dist_coefs)
            # uv = uv / 1.
            # u = np.round(uv[..., 0]).astype(int)
            # v = np.round(uv[..., 1]).astype(int)
    
            # valid_indices = np.logical_and.reduce([u >= 0, u < w, v >= 0, v < h])

            # u_valid = u[valid_indices]
            # v_valid = v[valid_indices]

            # result[v_valid, u_valid] = 1
            # results[c,:,:] = result
            # c+=1

        # tqdm_bar.set_description(f"Currently {len(imgs_hit)} images hit...")
    
    if save_points:
        if not path_to_masks:
            save_point_dir = save_points.rsplit('/',1)[0]
            if not os.path.exists(save_point_dir):
                os.makedirs(save_point_dir)
            np.save(save_points,np.asarray(comp_list3d,dtype=object))

    return comp_list3d if path_to_masks else results


# def filter_points(comp_list3d,threshold_count,save_points=""):
#     # Flatten the nested list into a single list of points
#     flattened_points = [point for sublist in comp_list3d for point in sublist]

#     # Convert points to tuples for counting
#     point_tuples = list(map(tuple, flattened_points))

#     # Count the occurrence of each point
#     point_counts = Counter(point_tuples)

#     # Filter points based on the threshold count and remove recurring entries
#     filtered_points = []
#     added_points = set()
#     for sublist in tqdm(comp_list3d,mininterval=10):
#         unique_sublist = []
#         for point in sublist:
#             if point_counts[tuple(point)] > threshold_count and tuple(point) not in added_points:
#                 unique_sublist.append(point)
#                 added_points.add(tuple(point))
#         if unique_sublist:
#             filtered_points.append(unique_sublist)

#     filtered_list = [item for sublist in filtered_points for item in sublist]

#     if save_points:
#         save_point_dir = save_points.rsplit('/',1)[0]
#         if not os.path.exists(save_point_dir):
#             os.makedirs(save_point_dir)
#         np.save(save_points,np.asarray(filtered_list,dtype=object))


#     return filtered_list

def filter_points(comp_list3d, threshold_count, save_points=""):
    #Count the occurrence of each point
    point_dict = {}
    filtered_list = []

    for sublist in tqdm(comp_list3d, mininterval=10):
        for point_ar in sublist:
            point = tuple(point_ar)
            if not point_dict.get(point):
                point_dict[point] = 1
            else:
                point_dict[point] += 1
    
    for item in point_dict:
        if point_dict[item] >= threshold_count:
            filtered_list.append(list(item))
    
    if save_points:
        if save_points[-1] =='/':
            save_points = save_points[:-1]
        save_point_dir = save_points.rsplit('/', 1)[0]
        if not os.path.exists(save_point_dir):
            os.makedirs(save_point_dir)
        np.save(save_points, np.asarray(filtered_list))

    return filtered_list
    # #Iterate through the nested list and count the occurrences of each point
    # point_counts = {}
    # if threshold_count >= 0:
    #     for sublist in tqdm(comp_list3d,mininterval=20):
    #         for point in sublist:
    #             if tuple(point) in point_counts:
    #                 point_counts[tuple(point)] += 1
    #             else:
    #                 point_counts[tuple(point)] = 1  
    #     filtered_list = [point for point, count in point_counts.items() if count >= threshold_count]
    
    
#     if save_points:
#         save_point_dir = save_points.rsplit('/', 1)[0]
#         if not os.path.exists(save_point_dir):
#             os.makedirs(save_point_dir)
#         np.save(save_points, np.asarray(filtered_list))

# #     return filtered_list

#     return [item for sublist in filtered_list for item in sublist]


def plot_results(results, transforms_path, path_to_masks,start=0,stop=0):

    with open(transforms_path) as f:
        transforms_json = json.load(f)
    transforms_json['frames'] = sorted(transforms_json['frames'],
                                            key=lambda x: int(x["file_path"].split("/")[-1].split(".")[0]))

    if not stop:
        stop = len(results)
    for ind,r in enumerate(results):
        if ind > start and ind < stop:
            if r.sum()>0:
                path_mask = path_to_masks + transforms_json['frames'][ind]['file_path'].split('/')[-1].replace('.jpg', '.png')
                img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
                plt.imshow(img_mask)
                plt.show()
                kernel = np.ones((10, 10), np.uint8)
                e = cv2.dilate(r, kernel)

                plt.imshow(e*255)
                plt.show()

    return None

def save_results(results, transforms_path, save_directory,save_directory_img,img_path, synth_method,test_masks_path):

    if synth_method == "NERF":
        with open(transforms_path) as f:
            transforms_json = json.load(f)
        transforms_json['frames'] = sorted(transforms_json['frames'],
                                                key=lambda x: int(x["file_path"].split("/")[-1].split(".")[0]))


        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if not os.path.exists(save_directory_img):
            os.makedirs(save_directory_img)

        if save_directory[-1] != '/':
            save_directory += '/'
        if save_directory_img[-1] != '/':
            save_directory_img += '/'
        if img_path[-1] != '/':
            img_path += '/'


        for ind,r in enumerate(results):
            #no '/' after save_dir pls... add self, i know sloppy, better use os path join
            out_path_mask = save_directory + transforms_json['frames'][ind]['file_path'].split('/')[-1].replace('.jpg', '.png')        
            out_path_img = save_directory_img + transforms_json['frames'][ind]['file_path'].split('/')[-1]        

            input_img = cv2.imread(img_path + transforms_json['frames'][ind]['file_path'].split('/')[-1])

            kernel = np.ones((5, 5), np.uint8)
            e = (cv2.dilate(r, kernel)*255).astype('uint8')
            e = cv2.erode(e,kernel)        
            cv2.imwrite(out_path_mask,e)
            if input_img.shape[:2] != e.shape[:2]:
                E = cv2.resize(e, (input_img.shape[1],input_img.shape[0]))
                new_img = cv2.bitwise_and(input_img,input_img,mask=E)
                cv2.imwrite(out_path_img, new_img)
            else:
                new_img = cv2.bitwise_and(input_img,input_img,mask=e)
                cv2.imwrite(out_path_img, new_img)
    elif synth_method == "GS":
        with open(transforms_path) as f:
            transforms_json = json.load(f)

        test_imgs_num = 0
        for i in range(len(transforms_json)):
            if i % 8 == 0:
                test_imgs_num +=1

        transforms_json = transforms_json[test_imgs_num:]
        copy_GTs = transforms_json[:test_imgs_num]

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if not os.path.exists(save_directory_img):
            os.makedirs(save_directory_img)

        if save_directory[-1] != '/':
            save_directory += '/'
        if save_directory_img[-1] != '/':
            save_directory_img += '/'
        if img_path[-1] != '/':
            img_path += '/'


        for ind,r in enumerate(results):
            #no '/' after save_dir pls... add self, i know sloppy, better use os path join
            out_path_mask = save_directory + transforms_json[ind]['img_name'] + '.png'    
            out_path_img = save_directory_img + transforms_json[ind]['img_name'] + '.jpg' 

            input_img = cv2.imread(img_path + transforms_json[ind]['img_name'] + '.jpg')

            kernel = np.ones((5, 5), np.uint8)
            e = (cv2.dilate(r, kernel)*255).astype('uint8')
            e = cv2.erode(e,kernel)        
            cv2.imwrite(out_path_mask,e)
            # if input_img.shape[:2] != e.shape[:2]:
            #     E = cv2.resize(e, (input_img.shape[1],input_img.shape[0]))
            #     new_img = cv2.bitwise_and(input_img,input_img,mask=E)
            #     cv2.imwrite(out_path_img, new_img)
            # else:
            #     new_img = cv2.bitwise_and(input_img,input_img,mask=e)
            #     cv2.imwrite(out_path_img, new_img)
            e = e.astype('float') / 255
            input_img = input_img.astype('float') / 255
            for channel in range(3):
                input_img[:,:,channel] = input_img[:,:,channel] * e
            
            new_img = (input_img * 255).astype('uint8')

            cv2.imwrite(out_path_img, new_img)
            print(f'out_imnames:{out_path_img}')
            
        
        # for t_im_data in copy_GTs:
        #     print(t_im_data['img_name'])

        #     out_path_mask_path = save_directory + t_im_data['img_name'] + '.png'
        #     out_path_img_path = save_directory_img + t_im_data['img_name'] + '.jpg'

        #     input_img_path = img_path + t_im_data['img_name'] + '.jpg'
        #     input_mask_path = test_masks_path + t_im_data['img_name'] + '.png'

        #     shutil.copy(input_img_path,out_path_img_path)
        #     shutil.copy(input_mask_path,out_path_mask_path)


def MVG_proj(list3d, json_path, path_to_masks,synth_method):

    if synth_method == "NERF":
        with open(json_path) as f:
            cam_json = json.load(f)
        
        print(json_path)

        #sort frames by order, 0000 first...
        cam_json['frames'] = sorted(cam_json['frames'], key=lambda x: int(x["file_path"].split("/")[-1].split(".")[0]))

        C_x = cam_json['cx']  # int(img1.shape[0]/2)
        C_y = cam_json['cy']  # int(img1.shape[1]/2)
        f_x = cam_json['fl_x']
        f_y = cam_json['fl_y']


        int_matrix = np.array([[-f_x, 0, C_x],
                        [0, f_y, C_y],
                        [0, 0, 1]])

        c = 0
        dist_coefs = None #np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)
        tqdm_bar = tqdm(cam_json['frames'])
        inside_mask_dict = {tuple(point): 0 for point in list3d}
        inside_image_list = np.zeros(len(list3d))
        counter = Counter(inside_mask_dict)


        for im_data in tqdm_bar:
            transf_matrix = np.array(im_data['transform_matrix'])

            transf_matrix_inv = np.linalg.inv(transf_matrix)

            R = transf_matrix_inv[:3,:3]
            t = transf_matrix_inv[:3,3]

            rvec, _ = cv2.Rodrigues(R)
            cam_pos = t

            path_mask = path_to_masks +  im_data['file_path'].split('/')[-1].replace('.jpg', '.png') 
            img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) # 
            #img_mask = cv2.undistort(img_mask,int_matrix,dist_coefs)
            
            h, w = img_mask.shape
            fast_uv = cv2.projectPoints(np.asarray(list3d), rvec, t, int_matrix, dist_coefs)

            fast_uv = np.asarray(np.round(fast_uv[0].reshape(len(fast_uv[0]),2)),dtype='int')
            
            #tried to speed code up by avoiding using any for loops
            inside_img = np.all([fast_uv[:,0] >= 0,fast_uv[:,0] < w,0 <= fast_uv[:,1],fast_uv[:,1] < h], axis=0)
            inside_frame_uv = fast_uv[inside_img] 
            inside_frame_xyz = list3d[inside_img] 
            
            inside_image_list += inside_img

            inside_mask = img_mask[inside_frame_uv[:,1],inside_frame_uv[:,0]] > 0
            inside_mask_xyz = inside_frame_xyz[inside_mask]

            inside_mask_xyz = tuple(map(tuple, inside_mask_xyz))
            counter.update(inside_mask_xyz)
            
        inside_mask_dict = dict(counter)

        inside_img_dict = {}
        for x in range(len(list3d)):
            inside_img_dict[tuple(list3d[x])] = inside_image_list[x]
        
        return inside_mask_dict, inside_img_dict
        
    elif synth_method == "GS": 
        with open(json_path) as f:
            cam_json = json.load(f)

        test_imgs_num = 0
        for i in range(len(cam_json)):
            if i % 8 == 0:
                test_imgs_num +=1

        cam_json = cam_json[test_imgs_num:]

        c = 0
        dist_coefs = None #np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)
        tqdm_bar = tqdm(cam_json) #tqdm(colmap_transforms_json['frames'])
        inside_mask_dict = {tuple(point): 0 for point in list3d}
        inside_image_list = np.zeros(len(list3d))
        counter = Counter(inside_mask_dict)
        
        for im_data in tqdm_bar:
            c += 1

            int_matrix = np.array([[im_data['fx'], 0, im_data['width']/2],
                    [0, im_data['fy'], im_data['height']/2],
                    [0, 0, 1]])

            R_gaussian = np.asarray(im_data['rotation'])
            T_gaussian = np.asarray(im_data['position'])
            t_mat = np.hstack((R_gaussian, T_gaussian.reshape(-1, 1)))
            transf_matrix = np.vstack((t_mat,[0,0,0,1]))
            transf_matrix_inv = np.linalg.inv(transf_matrix)

            R = transf_matrix_inv[:3,:3]
            t = transf_matrix_inv[:3,3]

            rvec, _ = cv2.Rodrigues(R)
            cam_pos = t
            
            path_mask = path_to_masks + im_data['img_name'] + '.png' #im_data['file_path'].split('/')[-1].replace('.jpg', '.png') 
            img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE) # 
            #img_mask = cv2.undistort(img_mask,int_matrix,dist_coefs)
            
            h, w = img_mask.shape
            fast_uv = cv2.projectPoints(np.asarray(list3d), rvec, t, int_matrix, dist_coefs)

            fast_uv = np.asarray(np.round(fast_uv[0].reshape(len(fast_uv[0]),2)),dtype='int')

            inside_img = np.all([fast_uv[:,0] >= 0,fast_uv[:,0] < w,0 <= fast_uv[:,1],fast_uv[:,1] < h], axis=0)
            inside_frame_uv = fast_uv[inside_img] 
            inside_frame_xyz = list3d[inside_img] 
            
            inside_image_list += inside_img

            inside_mask = img_mask[inside_frame_uv[:,1],inside_frame_uv[:,0]] > 0
            inside_mask_xyz = inside_frame_xyz[inside_mask]

            inside_mask_xyz = tuple(map(tuple, inside_mask_xyz))
            counter.update(inside_mask_xyz)
            
        inside_mask_dict = dict(counter)

        inside_img_dict = {}
        for x in range(len(list3d)):
            inside_img_dict[tuple(list3d[x])] = inside_image_list[x]
        
        return inside_mask_dict, inside_img_dict

    return None

    


def filter_fun(inside_mask_dict,inside_img_dict,list3d, thresh_coef,tot_frames_for_no_norm=0):
    filtered_list = []
    if tot_frames_for_no_norm > 0:
        print(tot_frames_for_no_norm)
        for cand_point in list3d:
            num_inside_image = inside_img_dict.get(tuple(cand_point))
            if num_inside_image:
                if float(num_inside_image)/tot_frames_for_no_norm >= thresh_coef:
                    filtered_list.append(cand_point)
    else:
        for cand_point in list3d:
            num_inside_mask = inside_mask_dict.get(tuple(cand_point))
            num_inside_image = inside_img_dict.get(tuple(cand_point))
            
            if num_inside_image:
                val =  num_inside_mask/num_inside_image #if normalize, then divide with num_inside_mask, else divide with 
            else:
                val = 0
            
            if val >= thresh_coef:
                filtered_list.append(cand_point)
    
    
    filtered_array = np.asarray(filtered_list,dtype='float')

    return filtered_array


def project_points_GS(transforms_path, list3d, path_to_masks="", save_points=""):
    """
    function for projecting points.

    transforms_path is path to transforms.json with camera poses in nerf coordinate system
    list3d contains the 3d point cloud which are input candidates for the reprojection
    path_to_masks is path to the masks for which we constrain the reprojection to be inside.

    returns a list containing all 3d points that were inside an image for each images when doing forward projection.
    """

    with open(transforms_path) as f:
        transforms_json = json.load(f)
    
    test_imgs_num = 0
    for i in range(len(transforms_json)):
        if i % 8 == 0:
            test_imgs_num +=1

    transforms_json = transforms_json[test_imgs_num:]

    comp_list3d = []

    dist_coefs = None # np.array([k1,k2,p1,p2,k3]); dist_coefs.shape = (5,1)

    h = transforms_json[0]['height']
    w = transforms_json[0]['width']
    #print("Intrinsic camera matrix: \n", int_matrix)
    tqdm_bar = tqdm(transforms_json) #tqdm(colmap_transforms_json['frames'])
    results = np.zeros((len(transforms_json), h,w),dtype='uint8')
    c=0
    for im_data in tqdm_bar:
        int_matrix = np.array([[im_data['fx'], 0, im_data['width']/2],
                [0, im_data['fy'], im_data['height']/2],
                [0, 0, 1]])

        R_gaussian = np.asarray(im_data['rotation'])
        T_gaussian = np.asarray(im_data['position'])
        t_mat = np.hstack((R_gaussian, T_gaussian.reshape(-1, 1)))
        transf_matrix = np.vstack((t_mat,[0,0,0,1]))
        transf_matrix_inv = np.linalg.inv(transf_matrix)

        R = transf_matrix_inv[:3,:3]
        t = transf_matrix_inv[:3,3]
        rvec, _ = cv2.Rodrigues(R)
        cam_pos = t
    
        list3d_hit = []
        
        if path_to_masks:
            print('dont call with masks...')

        else:
            result = np.zeros((h,w))
            p = np.expand_dims(list3d, axis=-1)
            uv, _ = cv2.projectPoints(p, rvec, t, int_matrix, dist_coefs)
            uv = uv / 1.
            u = np.round(uv[..., 0]).astype(int)
            v = np.round(uv[..., 1]).astype(int)
    
            valid_indices = np.logical_and.reduce([u >= 0, u < w, v >= 0, v < h])

            u_valid = u[valid_indices]
            v_valid = v[valid_indices]

            result[v_valid, u_valid] = 1
            results[c,:,:] = result
            c+=1

        # tqdm_bar.set_description(f"Currently {len(imgs_hit)} images hit...")
    
    if save_points:
        if not path_to_masks:
            save_point_dir = save_points.rsplit('/',1)[0]
            if not os.path.exists(save_point_dir):
                os.makedirs(save_point_dir)
            np.save(save_points,np.asarray(comp_list3d,dtype=object))

    return comp_list3d if path_to_masks else results


def NN_add_points(points, k, max_dist):

    # Build a KD-tree for fast nearest neighbor search
    kdtree = cKDTree(points)

    neighbors = [[] for x in range(len(points))]
    new_points = []
    for i, point in tqdm(enumerate(points)):
        # Query the KD-tree for points within the radius
        distance, neigh = kdtree.query(point, k=k)
        for n in range(1,k):
            if distance[n] > max_dist:
                new_point = (points[neigh[n]] + point)/2
                new_points.append(new_point)
    
    return np.append(points,np.unique(new_points,axis=0),axis=0)