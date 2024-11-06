import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
# from sklearn.cluster import DBSCAN
import builtins
def disable_print():
    builtins.print = lambda *args, **kwargs: None
class WindowSeg:

    def __init__(self,input_data):
        #原始图像
    
        self.img=input_data
        if self.img is None:
            raise ValueError(f"输入的图像数组为空！")
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.img_copy=self.img.copy()
    
        #灰度图像
        self.gray= None
        #模糊图像
        self.blurred = None
        self.gaussianBlur=None
        self.medianBlur=None
        self.bilateralFilter=None
        #光照补偿图像
        self.lightCompensation=None
       
        #边缘检测图像
        self.edges = None   
        #均衡化图像
        self.equalized = None
        #全局均值化图像
        self.thresh_otsu= None

        #翻转后的图像
        self.thresh_otsu_inverted = None
        #形态学去噪后的图像
        self.opening = None
        #分水岭算法图像
        self.background = None
        self.frontground = None
        self.unknown = None
        self.markers = None
        self.mask = None    
        self.binary_semantic_map=None

        self.num_labels=None
        self.labels=None
        self.stats=None
        self.centroids=None
        self.area_total=None
        self.area_threshold=None
    
        self.refine_stats=[]
        self.refine_centroids=[]

        self.rectangles=[]
        self.x_idGroup=[]
        self.y_idGroup=[]

        #可视化图片
        self.refine_map=None

        #合并图片
        self.img_merge=None

    def img_preprocess(self)->None:
        # self.gaussianBlur = cv2.GaussianBlur(self.img, (3, 3), 0)
        # self.medianBlur = cv2.medianBlur(self.img, 5)
        self.bilateralFilter = cv2.bilateralFilter(self.img, 5, 75, 75)

        self.blurred=self.bilateralFilter
        self.gray = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2GRAY)
        self.edges=cv2.Canny(self.gray, 100, 200)
                
        # 计算梯度
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 计算梯度百分位阈值，例如选取前10%的区域
        percentile = 90  # 保留前10%最大梯度区域
        gradient_threshold = np.percentile(gradient_magnitude, percentile)

        # 筛选梯度大于阈值的区域
        strong_gradients = gradient_magnitude > gradient_threshold

        # 提取局部区域
        local_region = self.img[strong_gradients]

        # 使用 Otsu 方法计算局部区域的阈值
        if len(local_region) > 0:
            otsu_threshold, _ = cv2.threshold(local_region.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            otsu_threshold = 0  # 如果没有局部区域，默认阈值为0

        # 应用 Otsu 阈值
        _, self.thresh_otsu = cv2.threshold(self.gray, otsu_threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
        # self.equalized = cv2.equalizeHist(self.blurred)
        # _, self.thresh_otsu = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # self.thresh_otsu = cv2.adaptiveThreshold(
        #     self.gray,                        # 输入灰度图像
        #     255,                              # 最大值
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # 高斯加权自适应阈值
        #     cv2.THRESH_BINARY,                # 二值化类型
        #     55,                               # 邻域大小，必须是奇数
        #     2                                 # 常数C，减去的值
        # )




    #计算窗户颜色分布 并判断是否需要反转颜色
    def calculate_color_distribution(self)->None:
        sum_x = np.sum(self.thresh_otsu, axis=0)  # 对每一列求和 (y方向)
        sum_y = np.sum(self.thresh_otsu, axis=1)  # 对每一行求和 (x方向)
        sum_x = sum_x /self.height
        sum_y = sum_y / self.width
        count_x=np.sum(sum_x<127)
        count_y=np.sum(sum_y<127)

        self.thresh_otsu_inverted = self.thresh_otsu
        if(count_x>self.width/2 and count_y>self.height/2):
            print("x方向和y方向的投影黑色居多，识别到玻璃为黑色，需要反转颜色")
            self.thresh_otsu_inverted = cv2.bitwise_not(self.thresh_otsu_inverted)
        elif(count_x>self.width/2 or count_y>self.height/2):
            print("窗户图像分割不佳，请调整参数或换张图片,退出程序......")
            # raise ValueError("窗户图像分割不佳，请调整参数或换张图片")
    #使用形态学变换去噪声
    def apply_morphological_denoising(self)->None:
        # 形态学去噪的尺寸需要进行调整 小图像需要小尺寸  大图像需要大尺寸 
        # 测试后发现3*3的kernel效果最好
        kernel = np.ones((3, 3), np.uint8)
        self.opening = cv2.morphologyEx(self.thresh_otsu_inverted, cv2.MORPH_OPEN, kernel, iterations=2)
    #使用分水岭算法
    def apply_watershed_algorithm(self)->None:
        self.frontground=self.opening
        self.background=self.opening
        self.unknown=cv2.subtract(self.background,self.frontground)
        #标记标签
        ret, self.markers = cv2.connectedComponents(self.frontground)
        # 增加1所有的背景区域变为0，以确保背景不是0, 而是1
        self.markers = self.markers + 1

        #标记未知区域
        self.markers[self.unknown == 255] = 0
        # self.markers = cv2.watershed(self.img, self.markers)
        #标记轮廓
        self.img[self.markers == -1] = [255, 0, 0]  

        self.mask = np.zeros((self.height,self.width), dtype=np.uint8)
        #标记掩膜为白色
        for i in range(2, self.markers.max() + 1):
            self.mask[self.markers == i] = 255
        #生成二值图
        self.binary_semantic_map=np.zeros((self.height, self.width), dtype=np.uint8)

        self.num_labels, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
        

        #设定面积阈值
        self.area_threshold=self.width*self.height*0.05
        for i in range(1, self.num_labels):  # 从1开始，因为0是背景
            x, y, w, h, area = self.stats[i]
            cx, cy = self.centroids[i]
            mask=np.zeros((self.height, self.width), dtype= bool)
            mask[y:y+h, x:x+w] = True
            if (area<self.area_threshold):
                # print(f"当前面积为{area}，小于面积阈值{self.area_threshold},忽略!!!")
                continue
            elif area>self.width*self.height*0.5:
                continue
            crop=self.img.copy()[y:y+h, x:x+w]
            self.rectangles.append(
                {
                    "id":i,
                    "stat":self.stats[i],
                    "mask":mask,
                    "centroid":self.centroids[i],
                    "crop":crop,
                }
            )
            self.binary_semantic_map[y:y+h, x:x+w]=i
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绘制矩形边界框
            # 绘制矩形中心
            center_x = int(cx)
            center_y = int(cy)
            cv2.circle(self.img, (center_x, center_y), 1, (0, 0, 255), -1)  # 绘制红色圆点标记中心
            cv2.putText(self.img,f"{i}",(center_x,center_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    def getPoints(self,x, y, w, h):
        points = []
        x, y, w, h = int(x), int(y), int(w), int(h)
        for i in range(x, x + w):
            points.append((i, y))
            points.append((i, y + h))
        for i in range(y, y + h):
            points.append((x, i))
            points.append((x + w, i))
        points.append((x+w,y+h))
        return points
    def pointToRect_distance(self,origin_point,points)->float:
        min_distance = float('inf')
        for point in points:
            distance=(origin_point[0]-point[0])**2+(origin_point[1]-point[1])**2
            min_distance=min(min_distance,distance)
        return min_distance
    def distance_min(self,box)->int:
        x, y, w, h = box
        print(f"当前矩形框为：{x,y,w,h}")
        # x, y, w, h = int(x), int(y), int(w), int(h)
        points = self.getPoints(x, y, w, h)
        total_area = np.sum(self.stats[:, 4])  # 提取第五列并计算总和
        total_distance = 0
        for i in range(self.num_labels):
            _, _, _, _, area = self.stats[i]
            cx, cy = self.centroids[i]
            # cx, cy = int(cx), int(cy)
            K=area/total_area
            distance = self.pointToRect_distance((cx, cy), points)*K
            total_distance += distance
        print(f"当前距离为：{total_distance}")
        return total_distance
    


    # 点到矩形边的距离计算函数
    def point_to_rect_distance(self,point, rect):
        x, y, w, h = rect
        px, py = point
        
        # 计算点到矩形四条边的距离
        left_distance = px - x
        right_distance = (x + w) - px
        top_distance = py - y
        bottom_distance = (y + h) - py
        
        # 如果点在矩形内部，返回到边界的最小距离
        if x <= px <= x + w and y <= py <= y + h:
            return min(left_distance, right_distance, top_distance, bottom_distance)
        
        # 如果点在矩形外部，计算点到矩形的欧氏距离
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        return np.sqrt(dx**2 + dy**2)

    # 总距离的目标函数
    def total_distance(self,rect, points):
        area_total = np.sum(self.refine_stats[: 4])  # 提取第五列并计算总和
        total_distance = 0
        for stat, centroid in zip(self.refine_stats, self.refine_centroids):
            x, y, w, h, area = stat
            cx, cy = centroid
            K=area / area_total
            distance = self.point_to_rect_distance((cx, cy), rect)*K
            total_distance += distance
        return total_distance


    def filter_rectangles(self)->None:
        if self.stats is None or self.centroids is None:
            return
        # for i in range(1, self.num_labels):
        #     x, y, w, h, area = self.stats[i]
        #     if area<self.width*self.height*0.5:
        #         self.refine_stats.append(self.stats[i])
        #         self.refine_centroids.append(self.centroids[i])
        
        # print(f"stats后:{self.refine_stats}")    
        # print(f"centroids后:{self.refine_centroids}")    
            
        area_max = np.max(self.stats[: 4])  # 提取第五列并计算总和
        width_max = np.max(self.stats[:, 2])
        height_max = np.max(self.stats[:, 3])
        array=[]
        
        for stat, centroid in zip(self.stats, self.centroids):
            x, y, w, h, area = stat
            cx, cy = centroid
            S=min(w/h,h/w)
            area=area/area_max
            w=w/width_max
            h=h/height_max
            array.append((w,h))
        # 获取排序后数组的下标，选择最后3个元素的下标（最大的数）
        # print(f"array为：{array}")
        print("array",array)
        # 绘制数组值的条形图

        data=np.array(array)
        if data.ndim == 1:
            return 
        if len(data) <= 1:
            return 
        # 使用 imshow 来展示二维数组
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(self.img)

        plt.subplot(1, 3, 2)
        for i in range(len(data)):
            plt.text(data[i,0], data[i,1], str(i), fontsize=12, ha='right')
        plt.scatter(data[:,0], data[:,1])
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('Scatter Plot of 2D Array')
        plt.grid(True)
            
        plt.subplot(1, 3, 3)

        dbscan = DBSCAN(eps=0.1, min_samples=2)
        labels = dbscan.fit_predict(data)
        print(f"聚类标签为：{labels}")

        # 获取唯一的聚类标签
        unique_labels = set(labels)
        print(f"聚类标签为：{unique_labels}")
        # 绘制散点图
        for label in unique_labels:
            if label == -1:
                # -1 代表噪声点
                color = 'k'  # 黑色
                marker = 'x'
            else:
                color = plt.cm.jet(label / len(unique_labels))  # 其他类的颜色
                marker = 'o'
                
            # 绘制每个聚类的点
            plt.scatter(data[labels == label][:, 0], data[labels == label][:, 1],
                        label=f'Cluster {label}' if label != -1 else 'Noise',
                        color=color, marker=marker)

        # 添加索引标签
        for i in range(len(data)):
            plt.text(data[i, 0], data[i, 1], str(i), fontsize=12, ha='right')

        # 设置坐标轴标签和标题
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('DBSCAN Clustering of 2D Array')
        plt.grid(True)
        plt.legend()
        # plt.show()

        area_cluster=np.zeros(len(unique_labels))
        #统计每个类的总面积
        for label in unique_labels:
            if label == -1:
                
                continue
            total_area = 0
            for i in range(len(labels)):
                if labels[i] == label:
                    total_area += self.stats[i][4]
            area_cluster[label]=total_area
        #获取最大面积的类别
        argmax=np.argmax(area_cluster)
        for i in range(len(self.stats)):
            if labels[i]==argmax:
                self.refine_stats.append(self.stats[i])
                self.refine_centroids.append(self.centroids[i])
               
                
    #玻璃分组及规则化
    def window_grouping(self)->None:
        x_sort=sorted(self.rectangles, key=lambda x: x['centroid'][0])
        x_group=[]
        if len(x_sort)==0:
            return
        current_group=[x_sort[0]]
        x_offset=0.05
        x_threshold=self.width*x_offset
        print(f"当前x分组阈值为不超过高度的{x_offset}，即不超过{x_threshold:.0f}像素")
        # 遍历排序后的数据点进行分组
        for i in range(1, len(x_sort)):
            if abs(x_sort[i]['centroid'][0] - current_group[0]['centroid'][0]) <= x_threshold:
                # print(abs(x_sort[i]['centroid'][0] - current_group[0]['centroid'][0]))
                current_group.append(x_sort[i])
                
            else:
                # print(f"当前分组为：{current_group}")
                x_group.append(current_group.copy()) 
                current_group=[]
                current_group.append(x_sort[i])
        #如果最后一个分组不为空，则添加到分组列表中        
        if current_group:
            x_group.append(current_group.copy())
        print(f"共分为{len(x_group)}组")

        y_temp=[]
        for group in x_group:
            x_temp=[]
            width_temp=[]
            for x in group:
                x_temp.append(x['stat'][0])
                width_temp.append(x['stat'][2])
            x_average = int(np.mean(x_temp))
            width_average = int(np.mean(width_temp))
            for x in group:
                x['stat'][0]=x_average
                x['stat'][2]=width_average
                y_temp.append(x)

        for x in x_group:
            temp=[]
            for elem in x:
                temp.append(elem['id'])
            self.x_idGroup.append(tuple(temp))

        y_sort=sorted(y_temp, key=lambda x: x['centroid'][1])
        y_group=[]
        current_group=[]
        current_group.append(y_sort[0])
        y_offset=0.1
        y_threshold=self.height*y_offset
        print(f"当前x分组阈值为不超过高度的{y_offset}，即不超过{y_threshold}")
        # 遍历排序后的数据点进行分组
        for i in range(1, len(y_sort)):
            if abs(y_sort[i]['centroid'][1] - current_group[0]['centroid'][1]) <= y_threshold and abs(y_sort[i]["stat"][4]-current_group[0]["stat"][4])<=current_group[0]["stat"][4]*y_offset:
                current_group.append(y_sort[i])
                
            else:
                # print(f"当前分组为：{current_group}")
                y_group.append(current_group.copy())
                current_group=[]
                current_group.append(y_sort[i])
                
        if current_group:
            y_group.append(current_group.copy())
    
        for group in y_group:
            temp=[]
            for elem in group:
                temp.append(elem['id'])
            self.y_idGroup.append(tuple(temp))
        
        for group in y_group:
            y_temp=[]
            height_temp=[]
            for y in group:
                y_temp.append(y['stat'][1])
                height_temp.append(y['stat'][3])
            y_average = int(np.mean(y_temp))
            height_average = int(np.mean(height_temp))
            for y in group:
                y['stat'][1]=y_average
                y['stat'][3]=height_average
    def generate_semantic_map(self)->None:
        self.refine_map=np.zeros((self.height,self.width),dtype=np.uint8)
        for elem in self.rectangles:
            x, y, w, h, area = elem['stat']
            self.refine_map[y:y+h, x:x+w]=elem['id']    
    def save(self,output_path:str)->None:
        cv2.imwrite(output_path, self.img_merge)
    def saveTotxt(self,image_dict:dict)->None:
        for title,image_array in image_dict.items():
            np.savetxt(f'{title}.txt', image_array, fmt='%d', delimiter=',')
            print(f"{title}的txt文件已保存")


    def merge(self,*args)->None:
        # 初始化 self.img_merge 如果它还没有图像
        if not hasattr(self, 'img_merge') or self.img_merge is None or self.img_merge.size == 0:
            self.img_merge = np.array([])
        for arg in args:
            #二值图
            if np.max(arg) > 127 and len(arg.shape) == 2:
                arg = cv2.cvtColor(arg, cv2.COLOR_GRAY2BGR)
            #标签图
            elif len(arg.shape) == 2:
                unique_labels = np.unique(arg)
                color_map = {}
                # 为每个标签生成随机颜色
                for label in unique_labels:
                    if label == 0:
                        # continue  # 假设 0 是背景，不分配颜色
                        # color_map[label] = tuple([70,185,255])
                        color_map[label] = tuple([0,0,0])
                        continue
                    # color_map[label] = tuple(np.random.randint(128, 256, 3).tolist())
                    # color_map[label] = tuple([214,110,40])
                    color_map[label] = tuple([255,255,255])
                # 创建一个彩色图像来存储结果
                colored_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                # 根据标签值为每个像素分配颜色
                for label, color in color_map.items():
                    colored_map[arg == label] = color
                arg = colored_map
            # arg=cv2.cvtColor(arg, cv2.COLOR_BGR2RGB)
            # 如果 self.img_merge 为空，直接赋值为第一个非空图像
            if self.img_merge.size == 0:
                self.img_merge = arg
            else:
                # 如果 arg 不为空，则进行拼接
                if arg is not None and arg.size > 0:
                    self.img_merge = cv2.hconcat([self.img_merge, arg]) 
    def display_images(self,image_dict:dict)->None:
        ''' 输入的 image_dict 是一个字典，键是图像的标题，值是图像的数组
        如下所示：
        image_dict = {
            "原始图像": img,
            "灰度图像": gray
        }
        '''
        
        plt.figure(figsize=(12, 6))
        for i, (title, image) in enumerate(image_dict.items(), 1):
            plt.subplot(1, len(image_dict), i)
            # 如果图像是三通道的，使用 cv2.cvtColor 将 BGR 转换为 RGB
            if len(image.shape) == 3:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # 如果图像是单通道的，且像素值大于 127，使用灰度图显示
            elif np.max(image) > 127:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)

            plt.title(title)
            plt.axis('off')
            # plt.suptitle(f'kernel大小为{self.kernel_size}\n'
            # f'当前图像的宽为：{self.width},kernel占宽度的{self.kernel_size/self.width:.2%}\n'
            # f'当前图像的高度为：{self.height},kernel占高度的{self.kernel_size/self.height:.2%}', fontsize=16)
        plt.show()  
        
    def compute_time(self,func):
        start = datetime.now()
        func()
        end = datetime.now()
        print(f"函数{func.__name__}耗时：{end-start}")
    def OverlapProcessing(self):
        i=0
        # print(f"当前矩形框数量为：{len(self.rectangles)}")
        while i<len(self.rectangles):
            current_mask=self.rectangles[i]['mask']
            overlap_count = 0  # 当前掩码与其他掩码的重叠次数
            masks_to_remove = []  # 需要移除的掩码索引
            for j in range(len(self.rectangles)):
                if i == j:
                    continue  # 跳过自己
                other_mask = self.rectangles[j]['mask']
                overlap = np.logical_and(current_mask, other_mask)
                overlap_area = np.sum(overlap>0)
                if overlap_area > 0: # 如果有重叠
                    overlap_count += 1
                    current_area = np.sum(current_mask>0)
                    other_area = np.sum(other_mask>0)
                    if overlap_area/current_area>=overlap_area/other_area:
                        masks_to_remove.append(j)
                    else:
                        masks_to_remove.append(i)
                        break
            if overlap_count > 1: # 如果有多个重叠
                    masks_to_remove.append(i)
            masks_to_remove = list(set(masks_to_remove))  # 去重
            masks_to_remove.sort(reverse=True)  # 逆序删除，以避免索引问题
            for idx in masks_to_remove:
                del self.rectangles[idx]
            i += 1
        # print(f"处理后矩形框数量为：{len(self.rectangles)}")
    def main(self)->None:
      
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.img_preprocess()
        self.calculate_color_distribution()
        self.apply_morphological_denoising()
        self.apply_watershed_algorithm()
        # self.filter_rectangles()
        self.OverlapProcessing()
        self.window_grouping()
        self.generate_semantic_map()
       
        # self.display_images({
        #         "原始图像": self.img,
        #     })
        
        # 保存图片
        # self.merge(self.img,self.blurred,self.gray, self.thresh_otsu,self.edges,self.thresh_otsu_inverted,self.opening,self.binary_semantic_map, self.refine_map)
        self.merge(self.refine_map)


def get_all_files(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
def folder_processing(input_folder:str,output_folder:str)->None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = output_folder + timestamp
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_paths=get_all_files(input_folder)
    fps_group=[]
    for file_path in tqdm(file_paths,desc="正在处理文件"):
        basename = os.path.basename(file_path)
        output_path = os.path.join(output_folder,basename)
        window=WindowSeg(file_path)
        window.main()
        # window.display_images({
        #         f"{basename}": window.refine_map,
        #     })
        # fps_group.append(window.fps)
        # fps_array=np.array(fps_group)
        # fps_average=np.mean(fps_array)
        # tqdm.write(f"fps:{fps_average:.2f}")
        window.save(output_path)
        

if __name__ == '__main__':
    
    '''
    文件夹推理

    '''
    # disable_print()
    # folder_processing(input_folder="E:\hhhh\mesh\LD",output_folder="result")
    '''
    单图片推理
    '''
    window=WindowSeg(r"E:\hhhh\mesh\data\cmp_b0145_8_SwinIR.png")
    window.main()
    window.save("test.png")
    window.display_images({
        "原始图像": window.img_copy,
        "bilateralFilter":window.bilateralFilter,
        "灰度图像": window.gray,
        "边缘检测": window.edges,
        "全局均值化": window.thresh_otsu,
        "统一范式提取": window.thresh_otsu_inverted,
        "开运算去噪": window.opening,
        "连通组件分析算法": window.markers,
        "语义图": window.binary_semantic_map,
        "最终语义图": window.refine_map,
        # "二值图":window.rectangles[2]['mask'],
        # "合成图": window.img_merges
        })
    
