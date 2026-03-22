from importlib import import_module
from typing import List, Union

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

from ..data_classes.registered_point_cloud import RegisteredPointCloud


def visualize_images(titles:Union[List[str],str], 
                     images:Union[List[np.ndarray],np.ndarray],
                     height:int=5)->None:
    """Visualize multiple images in a single row.

    Args:
        titles (Union[List[str],str]): List of titles or title for each image.
        images (Union[List[np.ndarray],np.ndarray]): List of BGR images or image to display.
    """
    if isinstance(titles, str):
        titles = [titles]
    if isinstance(images, np.ndarray):
        images = [images]
    cols = len(images)
    xy_aspect = images[0].shape[1] / images[0].shape[0]
    col_size = height * xy_aspect
    fig, axes = plt.subplots(1, cols, figsize=(col_size * cols, height))
    if cols == 1:
        axes = [axes]
    
    # Ensure we have the same number of titles as images
    if len(titles) < len(images):
        titles = titles + [''] * (len(images) - len(titles))
    for ax, title, image in zip(axes, titles, images):
        if len(image.shape) == 2:
            ax.imshow(image, cmap='plasma')
        else:
            ax.imshow(image[..., ::-1])  # Convert BGR to RGB for displaying
        ax.set_title(title)
        ax.axis('off')
    plt.show()


def visualize_point_clouds(
    titles: List[str],
    point_clouds: List[Union[List[Union[str, np.ndarray]], str, np.ndarray]],
) -> None:
    """Visualize one or more point-cloud plots using the registered-point-cloud renderer.

    Args:
        titles (List[str]): Title for each plot column.
        point_clouds (List[Union[List[Union[str, np.ndarray]], str, np.ndarray]]):
            Point clouds for each plot column. If an element is a list, its point clouds
            are rendered together in one plot with different colors.
    """
    if len(titles) != len(point_clouds):
        raise ValueError("titles and point_clouds must have the same length.")
    if len(point_clouds) == 0:
        raise ValueError("point_clouds must contain at least one plot.")

    try:
        o3d = import_module("open3d")
    except ImportError as exc:
        raise ImportError(
            "visualize_point_clouds requires open3d. Install it with: pip install open3d"
        ) from exc

    rendered_images: List[np.ndarray] = []

    for plot_clouds in point_clouds:
        if isinstance(plot_clouds, list):
            clouds_in_plot = plot_clouds
        else:
            clouds_in_plot = [plot_clouds]

        loaded_clouds = []
        for cloud in clouds_in_plot:
            if isinstance(cloud, str):
                pcd = o3d.io.read_point_cloud(cloud)
                points = np.asarray(pcd.points, dtype=np.float32)[:, :3]
            else:
                points = np.asarray(cloud, dtype=np.float32)
                if points.ndim != 2 or points.shape[1] < 3:
                    raise ValueError(
                        "Each ndarray in point_clouds must have shape (N, 3) or greater."
                    )
                points = points[:, :3]

            if points.size == 0:
                continue

            loaded_clouds.append(points)

        if len(loaded_clouds) == 0:
            raise ValueError("Each plot in point_clouds must contain at least one valid point cloud.")

        transforms = [np.eye(4, dtype=np.float32) for _ in loaded_clouds]
        rendered = RegisteredPointCloud(
            data=loaded_clouds,
            translations=transforms,
        )._visualize()
        rendered_images.append(rendered)

    visualize_images(titles, rendered_images)


def show_3d_poses(poses_3d,clear=True):
    # H36M形式の関節の接続情報（インデックスで定義）
    connections = [
        (0, 1), (1, 2), (2, 3),  # 右脚
        (0, 4), (4, 5), (5, 6),  # 左脚
        (0, 7), (7, 8), (8, 9),(9, 10),  # 胴体と頭
        (8, 11), (11, 12), (12, 13),  # 左腕
        (8, 14), (14, 15), (15, 16)   # 右腕
    ]

    # 3Dプロットの作成
    deg=180
    for pose in np.array(poses_3d):
        if clear:
            clear_output(wait=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3Dプロット用にprojection='3d'を追加
        
        # 関節点をプロット
        ax.scatter( pose[:,2],pose[:,0], pose[:,1], color='blue')  # ｚ, x, yの順にプロット
        
        # 各関節を線で繋ぐ
        for k,(i, j) in enumerate(connections):
            if k < 3:
                color = "blue"  # 右脚
            elif k < 6:
                color = "red"  # 左脚
            elif k < 10:
                color = "black"  # 胴体と頭
            elif k < 13:
                color = "pink"  # 左腕
            else:
                color = "green"  # 右腕
            # 可視化で見やすくするため軸に対するデータのxyzを入れ替えている
            ax.plot([pose[i, 2], pose[j, 2]],  # Z軸の座標
                [pose[i, 0], pose[j, 0]],  # X軸の座標
                    [pose[i, 1], pose[j, 1]],  # Y軸の座標
                    color=color)
        
        # 軸の設定
        ax.set_xlim([-1, 1])
        ax.set_ylim([1, -1])
        ax.set_zlim([1, -1])
        # 可視化で見やすくするため軸に対するデータのxyzを入れ替えている
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('Y axis')
        

        ax.view_init(elev=20, azim=deg)
        # ax.axis("off")
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        # time.sleep(0.05)
        deg+=1%360
        
        plt.show()