from sklearn.cluster import DBSCAN
import numpy as np

# 窗口位置数据，已知的窗口水平位置
positions = np.array([[0], [1.5], [3.0], [4.5], [7]])

# 使用DBSCAN进行聚类，设定最大间距eps为1.5，至少2个点形成一个簇
clustering = DBSCAN(eps=1.5, min_samples=1).fit(positions)
labels = clustering.labels_

# 计算每个簇的中心位置
cluster_centers = []
for label in set(labels): 
    if label == -1:  # 忽略噪声点
        continue
    cluster_points = positions[labels == label]
    center = cluster_points.mean()  # 计算簇中心
    cluster_centers.append(center)

# 检查簇中心之间是否存在较大的空隙，来判断是否有缺失窗口
missing_positions = []
for i in range(len(cluster_centers) - 1):
    # 如果两个簇中心之间的距离超过一定阈值，则认为之间可能有缺失窗口
    if cluster_centers[i + 1] - cluster_centers[i] > 1.5 * 1.5:  # 可调节系数1.5
        # 将可能缺失窗口的位置添加到列表中
        missing_position = (cluster_centers[i] + cluster_centers[i + 1]) / 2
        missing_positions.append(missing_position)

print("Detected missing window positions:", missing_positions)
