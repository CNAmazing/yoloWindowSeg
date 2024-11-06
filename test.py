# -*- coding: utf-8 -*-
import os
from ultralytics import YOLOv10
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
from skimage.metrics import structural_similarity as ssim
from windowSeg import WindowSeg
'''
输入：预测的文件夹路径
输出：每张图片的plt展示结果
'''

def binarize_image(image):
    # 使用cv2将RGB图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度图像的全局均值
    mean_value = np.mean(gray_image)
    # 使用全局均值进行二值化
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    return binary_image

def calculate_symmetry_score(binary_image):
    # 确保宽度和高度为偶数，避免SSIM尺寸错误
    height, width = binary_image.shape
    if height % 2 != 0:
        binary_image = binary_image[:height - 1, :]
    if width % 2 != 0:
        binary_image = binary_image[:, :width - 1]

    # 将图像分成左右两半
    height, width = binary_image.shape
    left_half = binary_image[:, :width // 2]
    right_half = np.fliplr(binary_image[:, width // 2:])  # 右半部分左右翻转

    # 将图像分成上下两半
    top_half = binary_image[:height // 2, :]
    bottom_half = np.flipud(binary_image[height // 2:, :])  # 下半部分上下翻转

    # 计算左右对称性得分
    left_right_score = ssim(left_half, right_half)

    # 计算上下对称性得分
    top_bottom_score = ssim(top_half, bottom_half)

    # 综合对称性得分，取左右和上下对称性得分的平均值
    symmetry_score = (left_right_score + top_bottom_score) / 2
    return symmetry_score

def select_most_symmetric_image(mask_list):
    highest_score = 0
    most_symmetric_image = None
    binary_images = []
    scores = []

    for mask in mask_list:
        # 使用cv2读取图像
        image = mask
        binary_image=image
        # 进行二值化
        binary_image = binarize_image(image)
        binary_images.append(binary_image)
        score = 0
        # 计算对称性得分
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        if num_labels>2:
            score = calculate_symmetry_score(binary_image)
        scores.append(score)
        
        if score > highest_score:
            highest_score = score
            most_symmetric_image = binary_image

    return most_symmetric_image, highest_score, binary_images, scores

def get_gtMaskList(image_path:str,label_path:str)->None:
    image=cv2.imread(image_path)
    with open(label_path, 'r') as file:
        lines = file.readlines()  # 读取所有行
    labels = []
    for line in lines:
        # 按空格分割每一行，并将其转换为 float 类型
        values = list(map(float, line.strip().split()))
        labels.append(values)

    h,w,_=image.shape
    i=1
    mask_gt=np.zeros((h,w),dtype=int)
    
    for label in labels:
        centriod_x,centriod_y,box_w,box_h=label[1:5]
        box_w=int(w*box_w)
        box_h=int(h*box_h)
        box_x=int(centriod_x*w-box_w/2)
        box_y=int(centriod_y*h-box_h/2)
        # print(f"box_x:{box_x},box_y:{box_y},box_w:{box_w},box_h:{box_h}")
        mask_gt[box_y:box_y+box_h,box_x:box_x+box_w]=i
        i+=1
    return mask_gt
def visual_result(predicted_folder:str):
    all_list=os.listdir(predicted_folder)
    image_list = [f for f in all_list if os.path.isfile(os.path.join(predicted_folder, f))]
    basename_list=[os.path.splitext(file)[0] for file in image_list]
    for instance in basename_list:
        image_path=os.path.join(predicted_folder,instance+".jpg")
        label_path=os.path.join(predicted_folder,"labels\\"+instance+".txt")
        mask=get_gtMaskList(image_path,label_path)
        plt.figure(figsize=(12, 6))
        plt.imshow(mask)
        plt.axis('off')
        plt.gcf().canvas.manager.set_window_title(f'{instance}')
        plt.show()

if __name__ == '__main__':


    model = YOLOv10('runs/train/exp6/weights/last.pt')
    '''
    source:测试图像路径 单张图片路径or文件夹路径
    '''
    result=model.predict(
        source='data/YOLODataset/images/val/00644.jpg',  # 测试图像路径
        save=True,           # 保存预测后的图片
        save_txt=True,      # 保存预测结果为txt文件
        show_labels=False,   # 不在图像上显示标签
        line_width=1         # 设置边界框线条宽度
    )
    """
    result是一个列表
    信息保存在result[0]中
    具体结构信息可查看md.md文件
    """
    orig_img=result[0].orig_img
    orig_shape=result[0].orig_shape
    color={
        0:(36, 231, 253),
        1:(67, 214, 114),
        2:(141, 103, 48),
    }
    height,width=orig_shape
    sematic_map = np.zeros((height, width, 3), dtype=np.uint8)
    sematic_map[:] = [84,0,68]
    names=result[0].names
    boxes=result[0].boxes   
    # print(f"boxes:{boxes}")
    cls=boxes.cls
    xyxy=boxes.xyxy
    crop_group=[]
    mask_group=[]
    save_dir=result[0].save_dir
    for class_id,box in zip(cls,xyxy):
        id=int(class_id)
        # print(f"class_name:{names[id]}\n")

        x_min,y_min,x_max,y_max=box
        x_min=int(torch.round(x_min))
        y_min=int(torch.round(y_min))
        x_max=int(torch.round(x_max))
        y_max=int(torch.round(y_max))
        
        sematic_map[y_min:y_max,x_min:x_max]=color[id]
        crop=orig_img[y_min:y_max,x_min:x_max].copy()
        window=WindowSeg(crop)
        window.main()
        sematic_map[y_min:y_max,x_min:x_max]=window.img_merge
        mask=sematic_map[y_min:y_max,x_min:x_max]
        crop_group.append(crop)
        mask_group.append(mask)
    #尺寸分组  根据宽高尺寸进行分组
    sizeGroups=[]
    threshold=0.2
    for i,crop in enumerate(crop_group):
        if i==0:
            h,w,_=crop.shape
            
            sizeGroups.append([(h,w,i)])
            continue
        h,w,_=crop.shape
        for group in sizeGroups:
            group_mean=np.mean(group,axis=0)
            h_mean,w_mean,_=group_mean
            if (abs(h-h_mean)<h_mean*threshold and abs(w-w_mean)<w_mean*threshold):
                group.append((h,w,i))
                break
            if group==sizeGroups[-1]:
                sizeGroups.append([(h,w,i)])
    """
    sizeGroups:结构
    sizeGroups[0]->分组1：[(h1,w1,i),(h2,w2,i),...]
    sizeGroups[2]->分组2：[(h1,w1,i),(h2,w2,i),...]
    """    
    print(f"sizeGroups:{sizeGroups}")
    #计算每个分组中的mask的对称性
    for group in sizeGroups:
        
        mask_list=[mask_group[i] for _,_,i in group]

        most_symmetric_image, highest_score, binary_images, scores = select_most_symmetric_image(mask_list)
        if most_symmetric_image is None:
            most_symmetric_image=np.zeros((50, 50, 1), dtype=np.uint8)
        print(f"scores:{scores}")
        print(f"highest_score:{highest_score}")
        for elem,score in zip(group,scores):
            _,_,i=elem
            box=xyxy[i]
            x_min,y_min,x_max,y_max=box
            x_min=int(torch.round(x_min))
            y_min=int(torch.round(y_min))
            x_max=int(torch.round(x_max))
            y_max=int(torch.round(y_max))
            cv2.putText(sematic_map, f"{score:.2f}", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        for h,w,i in group:
            
            new_mask=cv2.resize(most_symmetric_image,(w,h))
            mask_group[i]=new_mask

    # 重新将mask填充到sematic_map中
    # for i,mask in enumerate(mask_group):
    #     if names[int(cls[i])]=="balcony" or names[int(cls[i])]=="door":
    #         continue
    #     h,w,_=crop_group[i].shape
    #     x_min,y_min,x_max,y_max=xyxy[i]
    #     x_min=int(torch.round(x_min))
    #     y_min=int(torch.round(y_min))
    #     x_max=int(torch.round(x_max))
    #     y_max=int(torch.round(y_max))
    #     # mask=cv2.resize(mask,(x_max-x_min,y_max-y_min))
    #     rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #     sematic_map[y_min:y_max,x_min:x_max]=rgb_mask
    #保存mask图片
    # for i,mask in enumerate(mask_group):
    #     cv2.imwrite(os.path.join(save_dir,f"crop_{i}_mask.jpg"),mask)

    cv2.imwrite(os.path.join(save_dir,"sematic_map.png"), sematic_map)

    """
    测试指标
    """
    # visual_result(r'E:\hhhh\yolov10\runs\detect\predict16')
