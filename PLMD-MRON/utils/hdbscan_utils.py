import hdbscan
import numpy as np


class HdbscanCluster():
    def __init__(self):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,          # 每个簇至少包含 5 个点
            min_samples=6,               # 每个核心点至少需要 1 个邻居
            cluster_selection_epsilon=0.1, # 点之间的距离不超过 2 的区域才会被聚类
            metric='euclidean'           # 使用欧几里得距离
        )
    def predict(self, points):
        # # 准备数据
        # points = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])  # 替换为实际数据
        # 进行聚类
        clusters = self.clusterer.fit_predict(points)

        # 找到每个簇的中心点
        unique_clusters = np.unique(clusters)
        centers = []
        denses = []
        counts = []

        for cluster in unique_clusters:
            if cluster != -1:  # -1 表示噪声
                cluster_points = points[clusters == cluster]
                center = np.mean(cluster_points, axis=0)
                center = np.round(center).astype(int) 
                centers.append(center)

                # 计算簇中点的数量
                count = len(cluster_points)
                counts.append(count)
                
                # 计算密集程度（平均距离）
                distances = np.linalg.norm(cluster_points - center, axis=1)
                density = np.mean(distances)

                denses.append(density)

        # 输出结果
        # print("聚类中心点：", centers)
        return centers,counts,denses


