import numpy as np
from scipy.spatial import KDTree

def find_closest_color(pixel, color_pal):
        """
        找到与给定像素最接近的颜色在 color_pal 中的索引。
        
        :param pixel: 输入像素的 RGB 值 (tuple 或 list)
        :param color_pal: 颜色调色板，形状为 (n, 3) 的 NumPy 数组
        :return: 最接近颜色在 color_pal 中的索引
        """
        distances = np.sqrt(np.sum((color_pal - pixel) ** 2, axis=1))
        return np.argmin(distances)

def find_intersection(start, end, gx1, gx2, gy1, gy2):
    """
    计算线段与矩形边界的交点。
    
    参数:
        start (tuple): 线段起点 (x1, y1)。
        end (tuple): 线段终点 (x2, y2)。
        gx1, gx2, gy1, gy2 (float): 矩形边界。
    
    返回:
        tuple: 交点坐标，如果没有交点则返回 None。
    """
    x1, y1 = start
    x2, y2 = end
    
    # 计算线段的方向向量
    dx = x2 - x1
    dy = y2 - y1
    
    # 计算与矩形边界的交点
    def compute_intersection(x, y, dx, dy, boundary):
        t = (boundary - x) / dx if dx != 0 else np.inf
        return (x + t * dx, y + t * dy) if 0 <= t <= 1 else None
    
    # 检查与四条边的交点
    intersections = []
    for boundary in [gx1, gx2]:
        intersection = compute_intersection(x1, y1, dx, dy, boundary)
        if intersection and gy1 <= intersection[1] <= gy2:
            intersections.append(intersection)
    for boundary in [gy1, gy2]:
        intersection = compute_intersection(y1, x1, dy, dx, boundary)
        if intersection and gx1 <= intersection[1] <= gx2:
            intersections.append((intersection[1], intersection[0]))
    
    # 返回离起点最近的交点
    if intersections:
        distances = [np.linalg.norm(np.array(start) - np.array(p)) for p in intersections]
        return intersections[np.argmin(distances)]
    return None

def find_navigation_target(centers, num_points, cluster_density, gx1, gx2, gy1, gy2, start):
    """
    找到长期导航目标点。
    
    参数:
        centers (np.ndarray): 聚类中心点矩阵，形状为 (N, 2)。
        num_points (np.ndarray): 对应的聚类簇点数量，形状为 (N, 1)。
        cluster_density (np.ndarray): 对应的聚类簇密度，越小表示密度越大，形状为 (N, 1)。
        gx1, gx2, gy1, gy2 (float): 区间范围，gx1 < gx2，gy1 < gy2。
        start (tuple): 当前坐标 (start_x, start_y)。
    
    返回:
        tuple: 长期导航目标点。
    """
    # 筛选出在 [gx1, gx2] x [gy1, gy2] 区间内的点
    mask = (centers[:, 0] >= gx1) & (centers[:, 0] <= gx2) & \
        (centers[:, 1] >= gy1) & (centers[:, 1] <= gy2)
    filtered_centers = centers[mask]
    filtered_num_points = num_points[mask]
    filtered_density = cluster_density[mask]
    
    if len(filtered_centers) > 0:
        # 计算每个候选点的得分
        start = np.array(start)
        
        # 1. 密度得分（密度越大，得分越高）
        density_scores = 1 / (filtered_density + 1e-9)  # 避免除零
        
        # 2. 簇中点数量得分（数量越多，得分越高）
        num_points_scores = filtered_num_points
        
        # 3. 距离得分（距离越近，得分越高）
        distances = np.linalg.norm(filtered_centers - start, axis=1)
        distance_scores = 1 / (distances + 1e-9)  # 避免除零
        
        # 加权得分
        weight_density = 0.5  # 密度权重
        weight_num_points = 0.4  # 簇中点数量权重
        weight_distance = 0.1  # 距离权重
        
        total_scores = (
            weight_density * density_scores +
            weight_num_points * num_points_scores +
            weight_distance * distance_scores
        )
        
        # 找到得分最高的点
        best_index = np.argmax(total_scores)
        return tuple(filtered_centers[best_index])
    else:
        # 如果不存在满足条件的点，找到全局最近的点
        start = np.array(start)
        distances = np.linalg.norm(centers - start, axis=1)
        nearest_index = np.argmin(distances)
        nearest_center = tuple(centers[nearest_index])
        
        # 计算 start 到 nearest_center 的连线与边界的交点
        intersection = find_intersection(start, nearest_center, gx1, gx2, gy1, gy2)
        if intersection:
            return intersection
        else:
            print("No intersection is found, return to the nearest cluster center point.")
            return nearest_center

def remove_sparse_coordinates(coords, radius=2, min_neighbors=6):
    """
    删除所有稀疏坐标（周围半径内坐标数量 < min_neighbors）返回剩余坐标。
    
    参数:
        coords (list): 输入坐标列表，格式为 [[x1, y1], [x2, y2], ...]。
        radius (float): 邻域半径。
        min_neighbors (int): 最小邻域数量阈值。
    
    返回:
        list: 过滤后的坐标列表。
    """
    if len(coords) == 0:
        return coords
    
    # 将坐标转换为 NumPy 数组
    points = np.array(coords)
    
    # 构建 KDTree 加速邻域搜索
    tree = KDTree(points)
    
    # 查询每个点的半径邻域内的所有点（包括自身）
    neighbor_indices = tree.query_ball_point(points, r=radius)
    
    # 统计每个点的邻域数量（排除自身）
    neighbor_counts = np.array([len(indices) - 1 for indices in neighbor_indices])
    
    # 保留邻域数量 >= min_neighbors 的点
    mask = neighbor_counts >= min_neighbors
    filtered_points = points[mask]
    
    return filtered_points

from PIL import Image, ImageDraw, ImageFont
import os

def add_text_with_rounded_rectangle(
        img, 
        text, 
        font_size=24,
        font_color=(0, 0, 0),
        bg_color=(230, 216, 173),  # 浅蓝色
        # bg_color=(0,165,255),
        padding=20,
        corner_radius=20,
        position=(50, 50),  # 左上角坐标
        font_path=None
    ):
    """
    在图片上添加带圆角矩形背景的文字
    
    参数：
    - image_path: 输入图片路径
    - text: 要添加的文字
    - font_size: 字体大小（默认24）
    - font_color: 文字颜色（默认黑色）
    - bg_color: 背景颜色（默认浅蓝色）
    - padding: 文字与背景框的边距（默认20像素）
    - corner_radius: 圆角半径（默认20像素）
    - position: 文字框左上角坐标（默认(50,50)）
    - font_path: 自定义字体路径（默认使用系统字体）
    """
    
    # # 打开原始图片
    # img = Image.open(image_path).convert("RGBA")
    # width, height = img.size
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    

    # 计算文字包围盒
    try:
        # 新版Pillow使用textbbox
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # 旧版Pillow兼容方案
        text_width, text_height = draw.textsize(text)

    # 计算背景框坐标
    x0, y0 = position
    x1 = x0 + text_width + 2*padding
    y1 = y0 + text_height + 2*padding

    # 绘制圆角矩形背景
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=corner_radius,
        fill=bg_color
    )

    # 计算文字位置（居中）
    text_x = x0 + padding
    text_y = y0 + padding

    # 添加文字
    draw.text(
        (text_x, text_y), 
        text, 
        fill=font_color, 
        encoding='utf-8'
    )
    
    return img