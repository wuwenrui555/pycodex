# %%
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Circle
from pyqupath.tiff import TiffZarrReader
from scipy.optimize import minimize
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# %%


# 方法1：使用tab20颜色映射给每个polygon分配颜色
def plot_contours(
    gdf,
    cmap="tab20",
    plot_center=False,
    plot_id=True,
    plot_circle=True,
    merge_list=None,
    text_size=12,
    text_color="black",
    figsize=(10, 10),
):
    # 获取颜色映射
    n_polygons = len(gdf)

    # 为每个polygon分配颜色
    colors = []
    for i in range(n_polygons):
        color_idx = i % 20  # tab20有20种颜色，循环使用
        color = colormaps[cmap](color_idx)
        colors.append(to_hex(color))

    # 绘制
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每个polygon，使用分配的颜色
    gdf.plot(
        ax=ax,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # 可选：添加中心点
    if plot_center:
        gdf.set_geometry("center").plot(ax=ax, color="red", markersize=10, alpha=0.8)

    if plot_id:
        for idx, row in gdf.iterrows():
            ax.annotate(
                str(idx),
                (row.center_x, row.center_y),
                ha="center",
                va="center",
                fontsize=text_size,
                color=text_color,
                weight="bold",
            )

    if plot_circle:
        if merge_list is not None:
            flattened_merge_list = [
                idx for group in merge_list if len(group) > 1 for idx in group
            ]
        else:
            flattened_merge_list = []

        circles = []
        centers = []
        colors = []

        for idx, row in gdf.iterrows():
            circles.append(Circle((row["center_x"], row["center_y"]), row["radius"]))
            centers.append([row["center_x"], row["center_y"]])

            if idx in flattened_merge_list:
                colors.append("red")
            else:
                colors.append("green")

        merged_collection = PatchCollection(
            circles,
            facecolors="none",
            edgecolors=colors,
            linewidths=1,
            linestyles="-",
        )
        ax.add_collection(merged_collection)

    ax.invert_yaxis()
    plt.title(f"Filtered Contours (n={len(gdf)})")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    return fig, ax


# 使用


# 从contours创建GeoDataFrame
def contours_to_gdf(contours):
    data = []

    for i, contour in enumerate(contours):
        # 计算面积
        area = cv2.contourArea(contour)

        # 创建多边形几何
        if len(contour) >= 3:  # 至少需要3个点才能构成多边形
            # 将contour转换为shapely多边形
            contour_points = contour.reshape(-1, 2)
            polygon = Polygon(contour_points)
            (center_x, center_y), radius = find_minimum_enclosing_circle(contour_points)
            data.append(
                {
                    "geometry": polygon,
                    "center": Point(center_x, center_y),
                    "area": area,
                    "center_x": center_x,
                    "center_y": center_y,
                    "radius": radius,
                }
            )

    # 创建GeoDataFrame
    gdf = (
        gpd.GeoDataFrame(data)
        .sort_values(["center_x", "center_y"])
        .reset_index(drop=True)
    )
    return gdf


# 使用示例


def create_circular_kernel(radius):
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x * x + y * y <= radius * radius
    kernel = np.zeros((diameter, diameter), dtype=np.uint8)
    kernel[mask] = 1
    return kernel


def plot_merged_polygons(
    gdf_filtered,
    merged_results,
    figsize=(15, 12),
    title="Merged Polygons with Minimum Enclosing Circles",
):
    """
    绘制合并后的polygons和最小包围圆

    Parameters:
    -----------
    gdf_filtered : GeoDataFrame
        包含所有polygons的GeoDataFrame
    merged_results : list
        合并结果列表，每个元素包含merge_group, avg_center_x, avg_center_y, circle_center, circle_radius等信息
    figsize : tuple
        图形大小，默认(15, 12)
    title : str
        图形标题

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制所有原始polygons
    gdf_filtered.plot(
        ax=ax, color="lightblue", alpha=0.5, edgecolor="black", linewidth=0.5
    )

    # 绘制合并结果
    for i, result in enumerate(merged_results):
        merge_group = result["merge_group"]

        # 高亮要合并的polygons
        for idx in merge_group:
            gdf_filtered.loc[[idx]].plot(
                ax=ax, color="red", alpha=0.7, edgecolor="darkred", linewidth=2
            )

        # 绘制新的中心点
        ax.plot(
            result["avg_center_x"],
            result["avg_center_y"],
            "go",
            markersize=12,
            label=f"New Center {i + 1}",
        )

        # 绘制最小包围圆
        circle = plt.Circle(
            result["circle_center"],
            result["circle_radius"],
            fill=False,
            color="green",
            linewidth=3,
            linestyle="--",
            label=f"Min Circle {i + 1}",
        )
        ax.add_patch(circle)

        # 添加文本标注
        ax.annotate(
            f"Group {i + 1}\nRadius: {result['circle_radius']:.1f}",
            xy=result["circle_center"],
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax


# 使用示例
# 创建网格式排序
def create_grid_numbering(gdf, grid_size=100):
    # 计算网格坐标
    gdf["grid_x"] = (gdf["center_x"] // grid_size).astype(int)
    gdf["grid_y"] = (gdf["center_y"] // grid_size).astype(int)

    # 按网格排序：先按行（y），再按列（x）
    gdf_sorted = gdf.sort_values(["grid_y", "grid_x"]).reset_index(drop=True)

    # 删除临时列
    gdf_sorted = gdf_sorted.drop(["grid_x", "grid_y"], axis=1)

    return gdf_sorted


def calculate_optimal_grid_size(gdf):
    # 计算所有圆心之间的最小距离
    from scipy.spatial.distance import pdist

    points = gdf[["center_x", "center_y"]].values
    distances = pdist(points)

    # 使用最小距离的一定倍数作为网格大小
    min_distance = np.min(distances)
    optimal_grid_size = min_distance * 0.8  # 80% 的最小距离

    print(f"最小圆心距离: {min_distance:.2f}")
    print(f"建议的 grid_size: {optimal_grid_size:.2f}")

    return optimal_grid_size


def merge_contours_and_find_circles(gdf, merge_list, radius_expand=0, grid_size=None):
    """
    对merge_list中的polygons进行合并，对其他单独的polygons也找最小圆
    """
    merged_results = []

    # 收集所有在merge_list中的索引
    merged_indices = set()
    for merge_group in merge_list:
        merged_indices.update(merge_group)

    # 1. 处理merge_list中的polygons（合并）
    for merge_group in merge_list:
        polygons_to_merge = gdf[gdf.index.isin(merge_group)]
        if len(polygons_to_merge) == 0:
            continue

        # 合并所有polygon的几何形状
        merged_geometry = unary_union(polygons_to_merge.geometry.tolist())

        # 找到包围所有polygon的最小圆
        all_points = []
        for idx in merge_group:
            polygon = gdf.loc[idx, "geometry"]
            coords = list(polygon.exterior.coords)
            all_points.extend(coords)
        all_points = np.array(all_points)

        (center_x, center_y), radius = find_minimum_enclosing_circle(all_points)

        merged_results.append(
            {
                "merge_group": merge_group,
                "geometry": merged_geometry,
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "total_area": polygons_to_merge["area"].sum(),
                "is_merged": True,  # 标记这是合并的结果
            }
        )

    # 2. 处理不在merge_list中的单独polygons
    for idx, row in gdf.iterrows():
        if idx not in merged_indices:
            # 对单个polygon找最小圆
            polygon = row["geometry"]
            coords = list(polygon.exterior.coords)
            all_points = np.array(coords)

            (center_x, center_y), radius = find_minimum_enclosing_circle(all_points)

            merged_results.append(
                {
                    "merge_group": [idx],  # 单个polygon作为一个组
                    "geometry": polygon,
                    "center_x": center_x,
                    "center_y": center_y,
                    "radius": radius,
                    "total_area": row["area"],
                    "is_merged": False,  # 标记这是单个polygon
                }
            )

    gdf = (
        gpd.GeoDataFrame(merged_results)
        .sort_values(["center_x", "center_y"])
        .reset_index(drop=True)
    )
    gdf["radius"] = gdf["radius"] + radius_expand

    if grid_size is None:
        optimal_grid_size = calculate_optimal_grid_size(gdf)
    gdf = create_grid_numbering(gdf, grid_size=optimal_grid_size)

    total_count = len(gdf)
    num_digits = len(str(total_count))
    new_index = [f"c{i:0{num_digits}d}" for i in range(total_count)]
    gdf.index = new_index
    return gdf


def find_minimum_enclosing_circle(points):
    """
    找到包围所有点的最小圆
    """

    def circle_error(params):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        return np.max(distances - r) ** 2

    # 初始猜测：所有点的中心和最大距离
    center_x = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
    center_y = (np.min(points[:, 1]) + np.max(points[:, 1])) / 2
    max_dist = np.max(
        np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
    )

    initial_guess = [center_x, center_y, max_dist]

    # 优化
    result = minimize(circle_error, initial_guess, method="Nelder-Mead")

    return (result.x[0], result.x[1]), result.x[2]


def plot_all_polygons_with_circles(
    gdf_filtered, all_results, figsize=(15, 12), radius_expand=0
):
    """
    绘制所有polygons（包括合并的和单独的）及其最小圆
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制所有原始polygons（作为背景）
    gdf_filtered.plot(
        ax=ax, color="lightgray", alpha=0.3, edgecolor="black", linewidth=0.3
    )

    for i, result in enumerate(all_results):
        if result["is_merged"]:
            # 合并的polygons用红色高亮
            for idx in result["merge_group"]:
                gdf_filtered.loc[[idx]].plot(
                    ax=ax, color="red", alpha=0.7, edgecolor="darkred", linewidth=2
                )

            # 绘制合并后的中心点（绿色）
            ax.plot(result["avg_center_x"], result["avg_center_y"], "go", markersize=10)

            # 绘制最小包围圆（绿色虚线）
            circle = plt.Circle(
                result["circle_center"],
                result["circle_radius"] + radius_expand,
                fill=False,
                color="green",
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(circle)

            # 添加标注
            ax.annotate(
                f"Merged {result['merge_group']}\nR: {result['circle_radius']:.1f}",
                xy=result["circle_center"],
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        else:
            # 单独的polygons用蓝色
            idx = result["merge_group"][0]
            gdf_filtered.loc[[idx]].plot(
                ax=ax, color="lightblue", alpha=0.6, edgecolor="blue", linewidth=1
            )

            # 绘制中心点（蓝色）
            ax.plot(result["avg_center_x"], result["avg_center_y"], "bo", markersize=4)

            # 绘制最小包围圆（蓝色细线）
            circle = plt.Circle(
                result["circle_center"],
                result["circle_radius"] + radius_expand,
                fill=False,
                color="blue",
                linewidth=1,
                linestyle="-",
                alpha=0.5,
            )
            ax.add_patch(circle)

    ax.set_title(
        f"All Polygons with Minimum Enclosing Circles (Total: {len(all_results)})"
    )
    ax.axis("equal")
    ax.invert_yaxis()
    plt.tight_layout()

    return fig, ax


def plot_all_polygons_with_circles_ultra_fast(
    gdf_filtered, all_results, figsize=(10, 10), radius_expand=0, show_annotations=False
):
    """
    极速版本，可选择是否显示注释
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 背景polygons
    gdf_filtered.plot(
        ax=ax, color="lightgray", alpha=0.3, edgecolor="black", linewidth=0.3
    )

    # 分离数据
    merged_data = [r for r in all_results if r["is_merged"]]
    single_data = [r for r in all_results if not r["is_merged"]]

    # 批量处理合并的polygons
    if merged_data:
        merged_idx = []
        merged_circles = []
        merged_centers = []

        for result in merged_data:
            merged_idx.extend(result["merge_group"])
            merged_circles.append(
                Circle(result["circle_center"], result["circle_radius"] + radius_expand)
            )
            merged_centers.append(
                [result["circle_center"][0], result["circle_center"][1]]
            )

        # 绘制
        gdf_filtered.loc[merged_idx].plot(
            ax=ax, color="red", alpha=0.7, edgecolor="darkred", linewidth=2
        )

        merged_collection = PatchCollection(
            merged_circles,
            facecolors="none",
            edgecolors="green",
            linewidths=2,
            linestyles="--",
        )
        ax.add_collection(merged_collection)

        merged_centers = np.array(merged_centers)
        ax.scatter(
            merged_centers[:, 0], merged_centers[:, 1], c="green", s=100, marker="o"
        )

    # 批量处理单独的polygons
    if single_data:
        single_idx = []
        single_circles = []
        single_centers = []

        for result in single_data:
            single_idx.extend(result["merge_group"])
            single_circles.append(
                Circle(result["circle_center"], result["circle_radius"] + radius_expand)
            )
            single_centers.append(
                [result["circle_center"][0], result["circle_center"][1]]
            )

        # 绘制
        gdf_filtered.loc[single_idx].plot(
            ax=ax, color="lightblue", alpha=0.6, edgecolor="blue", linewidth=1
        )

        single_collection = PatchCollection(
            single_circles,
            facecolors="none",
            edgecolors="blue",
            linewidths=1,
            linestyles="-",
            alpha=0.5,
        )
        ax.add_collection(single_collection)

        single_centers = np.array(single_centers)
        ax.scatter(
            single_centers[:, 0], single_centers[:, 1], c="blue", s=16, marker="o"
        )

    # 可选注释
    if show_annotations:
        for result in merged_data:
            ax.annotate(
                f"Merged {result['merge_group']}\nR: {result['circle_radius']:.1f}",
                xy=result["circle_center"],
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

    ax.set_title(
        f"All Polygons with Minimum Enclosing Circles (Total: {len(all_results)})"
    )
    ax.axis("equal")
    ax.invert_yaxis()
    plt.tight_layout()

    return fig, ax


def find_circle_overlaps(idx, center_x, center_y, radius, radius_expand=0):
    """
    找出需要合并的圆的索引组合

    Args:
        gdf: GeoDataFrame with columns 'center_x', 'center_y', 'radius'

    Returns:
        List of tuples: [(group1_indices), (group2_indices), ...]
        例如: [(0, 14), (5, 8, 12), (20,)]
    """
    df = pd.DataFrame(
        {"idx": idx, "center_x": center_x, "center_y": center_y, "radius": radius}
    ).set_index("idx")

    if len(df) == 0:
        return []

    merge_list = []
    used = set()
    indices = list(df.index)  # 获取原始索引

    for idx in indices:
        if idx in used:
            continue

        # 开始一个新的合并组
        merge_group = [idx]
        used.add(idx)

        # 获取当前圆的信息
        current_row = df.loc[idx]

        # 找到所有与当前圆相交的圆
        for other_idx in indices:
            if other_idx in used:
                continue

            other_row = df.loc[other_idx]

            # 计算两个圆心之间的距离
            distance = np.sqrt(
                (current_row["center_x"] - other_row["center_x"]) ** 2
                + (current_row["center_y"] - other_row["center_y"]) ** 2
            )

            # 检查是否相交: distance < sum of radii
            if distance < (current_row["radius"] + other_row["radius"]):
                merge_group.append(other_idx)
                used.add(other_idx)

        # 将合并组添加到结果中
        if len(merge_group) > 1:
            merge_list.append(tuple(merge_group))

    return merge_list


def plot_contours_to_merge(gdf_filtered, merge_list, figsize=(10, 10)):
    flattened_merge_list = [
        idx for group in merge_list if len(group) > 1 for idx in group
    ]

    fig, ax = plt.subplots(figsize=figsize)

    # 背景polygons
    gdf_filtered.plot(
        ax=ax, color="lightblue", alpha=0.6, edgecolor="blue", linewidth=0.5
    )

    circles = []
    centers = []
    colors = []

    for idx, row in gdf_filtered.iterrows():
        circles.append(Circle((row["center_x"], row["center_y"]), row["radius"]))
        centers.append([row["center_x"], row["center_y"]])

        if idx in flattened_merge_list:
            colors.append("red")
        else:
            colors.append("green")

    merged_collection = PatchCollection(
        circles,
        facecolors="none",
        edgecolors=colors,
        linewidths=1,
        linestyles="-",
    )
    ax.add_collection(merged_collection)

    centers = np.array(centers)
    ax.scatter(centers[:, 0], centers[:, 1], c=colors, s=25, marker="o")

    ax.invert_yaxis()

    return fig, ax


def restore_to_original_scale(gdf, downsample_step=16):
    """
    将下采样图像中的圆心和半径还原到原始图像坐标系

    Args:
        gdf: 基于下采样图像的GeoDataFrame
        downsample_step: 下采样步长（倍数）

    Returns:
        GeoDataFrame: 还原到原始图像坐标系的数据
    """

    # 将坐标和半径都乘以下采样倍数
    names = gdf.index
    center_x_list = gdf["center_x"] * downsample_step
    center_y_list = gdf["center_y"] * downsample_step
    radius_list = gdf["radius"] * downsample_step

    geometry_list = []
    for center_x, center_y, radius in zip(center_x_list, center_y_list, radius_list):
        # 生成原始尺度的圆形
        angles = np.arange(0, 360, 1)
        angles_rad = np.radians(angles)

        x_points = center_x + radius * np.cos(angles_rad)
        y_points = center_y + radius * np.sin(angles_rad)

        circle_points = list(zip(x_points, y_points))
        circle_polygon = Polygon(circle_points)
        geometry_list.append(circle_polygon)

    return gpd.GeoDataFrame(
        {
            "geometry": geometry_list,
            "name": names,
            "center_x": center_x_list,
            "center_y": center_y_list,
            "radius": radius_list,
        }
    )


# %%


img_f = Path(
    "/mnt/nfs/storage/HBV-HCC/CODEX-images/formal/first_run/HCC_batch2_VICTORY_TMA_JAN1A2023_0.er.qptiff"
)
# img_f = Path(
#     "/mnt/nfs/home/huayingqiu/foundation/KRONOS_v2_QC/data/tma/21125_JS_PCNSL_TBVTMA_TMA1.er.qptiff"
# )
reader = TiffZarrReader.from_qptiff(img_f)
img_dapi = reader.zimg_dict["DAPI"][:, :]

# %%

downsample_step = 16
kernel_radius = 5
area_threshold = 1000

img = img_dapi[::downsample_step, ::downsample_step]

_, otsu = cv2.threshold(img, 0, 2**16 - 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = create_circular_kernel(kernel_radius)
otsu_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
mask_8bit = (otsu_close > 0).astype(np.uint8)

# Find contours in the binary mask
contours, hierarchy = cv2.findContours(
    mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

gdf = contours_to_gdf(contours)
gdf_filtered = gdf[gdf.area >= area_threshold]


merge_list = find_circle_overlaps(
    gdf_filtered.index,
    gdf_filtered["center_x"],
    gdf_filtered["center_y"],
    gdf_filtered["radius"],
)

fig, ax = plot_contours(gdf_filtered, merge_list=merge_list)
fig.show()

# %%
gdf_merge = merge_contours_and_find_circles(gdf_filtered, merge_list, radius_expand=5)

overlap_list = find_circle_overlaps(
    gdf_merge.index,
    gdf_merge["center_x"],
    gdf_merge["center_y"],
    gdf_merge["radius"],
)

fig, ax = plot_contours(gdf_merge, merge_list=overlap_list)
fig.show()

# %%

downsample_step = 16
gdf_merge

gdf_original = restore_to_original_scale(gdf_merge, downsample_step)
gdf_original
# %%
