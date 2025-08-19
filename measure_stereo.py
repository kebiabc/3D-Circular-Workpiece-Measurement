import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fit_circle(image_path):
    """从单张图像中拟合圆形轮廓并返回圆心和半径"""
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 高斯模糊去噪
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Canny 边缘检测
    edges = cv2.Canny(blur, 50, 150)
    
    # 轮廓提取
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # 寻找最大轮廓
    max_radius = 0
    best_contour = None
    best_center = None
    
    for c in contours:
        pts = np.array(c, dtype=np.float32).reshape(-1, 2)
        center, radius = cv2.minEnclosingCircle(pts)
        if radius > max_radius:
            max_radius = radius
            best_contour = c
            best_center = center
    
    # 亚像素优化
    if best_contour is None:
        raise ValueError("未检测到有效轮廓")
    
    pts = np.array(best_contour, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.cornerSubPix(img, pts, (5, 5), (-1, -1), criteria)
    pts = pts.reshape(-1, 2)
    center, radius = cv2.minEnclosingCircle(pts)
    
    # 可视化结果
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(output, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
    cv2.circle(output, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
    
    # 保存结果
    output_path = image_path.replace(".jpg", "_result.jpg")
    cv2.imwrite(output_path, output)
    
    print(f"图像 {image_path} 处理完成，结果保存至 {output_path}")
    return np.array(center), radius

def triangulate_points(point1, point2, P1, P2):
    """使用三角测量计算3D点位置"""
    # 构建齐次坐标
    points = np.zeros((2, 2))
    points[0, :] = point1
    points[1, :] = point2
    
    # 三角测量
    point_4d = cv2.triangulatePoints(P1, P2, points[0:1, :].T, points[1:2, :].T)
    
    # 转换为非齐次坐标
    point_3d = point_4d[:3] / point_4d[3]
    
    return point_3d.flatten()

def calculate_diameter(center_3d, radius_px1, radius_px2, K1, K2, R, T):
    """计算工件的实际直径"""
    # 计算相机1坐标系下的深度
    depth1 = center_3d[2]
    
    # 计算相机2坐标系下的深度
    # 将点转换到相机2坐标系
    center_cam2 = R @ center_3d + T
    depth2 = center_cam2[2]
    
    # 计算实际直径 (使用两个视图的平均值)
    fx1 = K1[0, 0]
    fx2 = K2[0, 0]
    
    diameter1 = (2 * radius_px1 * depth1) / fx1
    diameter2 = (2 * radius_px2 * depth2) / fx2
    
    return (diameter1 + diameter2) / 2.0, depth1

def visualize_3d_points(center_3d, diameter, depth, T):
    """可视化3D点"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制圆心
    ax.scatter(center_3d[0], center_3d[1], center_3d[2], c='r', s=100, label='Circle Center')
    
    # 创建表示工件的圆盘
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center_3d[0] + (diameter / 2) * np.cos(theta)
    y = center_3d[1] + (diameter / 2) * np.sin(theta)
    z = np.full_like(x, center_3d[2])
    
    ax.plot(x, y, z, c='b', label='Workpiece')
    
    # 设置坐标轴标签
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'3D Workpiece Reconstruction\nDiameter: {diameter:.2f}mm, Depth: {depth:.2f}mm')
    
    # 添加相机位置示意
    ax.scatter(0, 0, 0, c='g', s=100, marker='^', label='Camera 1')
    ax.scatter(T[0], T[1], T[2], c='m', s=100, marker='^', label='Camera 2')
    
    # 添加深度线
    ax.plot([center_3d[0], center_3d[0]], 
            [center_3d[1], center_3d[1]], 
            [0, center_3d[2]], 'k--', label=f'Depth: {depth:.2f}mm')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('3d_reconstruction.png')
    plt.show()

def main():
    # 示例相机参数 - 需要替换为实际标定参数
    # 相机1内参矩阵
    # 左相机内参
    fx_l = 6432.718156
    fy_l = 6433.640957
    cx_l = 2510.513808
    cy_l = 2540.649788

    # 右相机内参
    fx_r = 6464.159743
    fy_r = 6466.869928
    cx_r = 2534.501657
    cy_r = 2554.850023


    K1 = np.array([[fx_l, 0, cx_l],
                    [0, fy_l, cy_l],
                    [0, 0, 1]])
    # 右相机内参
    K2 = np.array([[fx_r, 0, cx_r],
                        [0, fy_r, cy_r],
                        [0, 0, 1]])
    
    # 相机1到相机2的旋转矩阵
    R = np.array([[0.999982515, 6.03612E-05, -0.005913132],
                        [-3.72253E-05, 0.999992345, 0.003912667],
                        [0.005913323, -0.003912378, 0.999974863]])
    
    # 相机1到相机2的平移向量 (单位：mm)
    T = np.array([53.68039957, -0.894192181, 0.882346604]) 
    
    # 畸变系数 (示例值)
    dist1 = np.array([-0.033465189, -0.2844377, 0.000308576, 0.00034481, 2.702545964])
    dist2 = np.array([-0.057723512, 0.375678793, -0.00025302, 5.63125E-05, -2.160836638]) 
    
    # 构建投影矩阵
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))
    
    # 处理两个视图
    try:
        center1, radius1 = fit_circle("tools/left.jpg")
        center2, radius2 = fit_circle("tools/right.jpg")
        
        print(f"视图1圆心: {center1}, 半径: {radius1} 像素")
        print(f"视图2圆心: {center2}, 半径: {radius2} 像素")
        
        # 三角测量得到3D点
        center_3d = triangulate_points(center1, center2, P1, P2)
        print(f"三维圆心坐标: {center_3d} mm")
        
        # 计算直径和深度
        diameter, depth = calculate_diameter(center_3d, radius1, radius2, K1, K2, R, T)
        print(f"工件直径: {diameter:.2f} mm")
        print(f"工件深度(Z坐标): {depth:.2f} mm")
        
        # 可视化结果
        visualize_3d_points(center_3d, diameter, depth, T)
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()