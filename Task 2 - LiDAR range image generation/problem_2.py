import numpy as np
import matplotlib.pyplot as plt

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj


def pointcloud_to_range_img(current_vertex, fov_up, fov_down, proj_H, proj_W, max_range):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
            current_vertex: raw point clouds
        Returns: 
            proj_range: projected range image with depth, each pixel contains the corresponding depth
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
    
    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]
    
    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    
    ############################# Problem ##############################
    
    # 1. LiDAR 센서 좌표계를 기준으로, 모든 point의 yaw 및 pitch 각도 구하기
    yaw = np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # 2. 3D point를 2D 이미지 좌표계로 투영했을 때의 픽셀 위치 구하기
    u = 0.5 * (1.0 - yaw / np.pi) * proj_W
    v = (1.0 - (pitch + abs(fov_down)) / fov) * proj_H
    
    ####################################################################
    
    
    # round and clamp for use as index
    u = np.floor(u)
    u = np.minimum(proj_W - 1, u)
    u = np.maximum(0, u).astype(np.int32)  # in [0,W-1]
    
    v = np.floor(v)
    v = np.minimum(proj_H - 1, v)
    v = np.maximum(0, v).astype(np.int32)  # in [0,H-1]
    
    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    u = u[order]
    v = v[order]

    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_range[v, u] = depth
    
    return proj_range




if __name__ == "__main__":
    fov_up=2.0 
    fov_down=-24.9 
    proj_H=64 
    proj_W=900

    lidar_points = load_from_bin('./0000000000.bin')
    
    range_img = pointcloud_to_range_img(lidar_points, fov_up, fov_down, proj_H, proj_W, max_range=150)

    # 정규화
    valid = range_img > 0
    range_img_vis = range_img.copy()
    range_img_vis[valid] = (range_img[valid] - range_img[valid].min()) / \
                           (range_img[valid].max() - range_img[valid].min())
    range_img_vis[~valid] = 0

    plt.subplots(1, 1, figsize=(13, 3))
    plt.title("Result of Vertical FOV (-24.9 , 2.0) & Horizontal FOV (-180 , 180)")
    plt.imshow(range_img_vis, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.show()