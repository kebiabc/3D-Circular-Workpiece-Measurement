import cv2
import numpy as np
from scipy.optimize import least_squares

# === 相机内参（需要先标定） ===
fx = 6350.955187
fy = 6352.45175107693
cx_cam = 2527.69608602686
cy_cam = 2545.5974390674

# 投影函数：物方圆 -> 像素轮廓
def project_circle(params, n_points):
    Xc, Yc, R, Z = params
    angles = np.linspace(0, 2*np.pi, n_points)
    pts_world = np.array([[Xc + R*np.cos(a), Yc + R*np.sin(a), Z] for a in angles])
    u = fx * (pts_world[:,0] / pts_world[:,2]) + cx_cam
    v = fy * (pts_world[:,1] / pts_world[:,2]) + cy_cam
    return np.vstack([u, v]).T

# 误差函数
def residuals(params, measured_pts):
    proj = project_circle(params, n_points=len(measured_pts))
    return (proj - measured_pts).ravel()

# 读取灰度图
img = cv2.imread("tools/6.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 提取轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if not contours:
    print("未找到轮廓！")
    exit()

# 找到最大的轮廓（面积最大）
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
pts = np.array(best_contour, dtype=np.float32).reshape(-1, 1, 2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
cv2.cornerSubPix(img, pts, (5,5), (-1,-1), criteria)
pts = pts.reshape(-1, 2)

# 粗拟合像素圆
(px, py), r_px = cv2.minEnclosingCircle(pts)

# 初值：像素换算粗估真实半径
Z_init = 200.0  # 物距初值
Xc_init = (px - cx_cam) * Z_init / fx
Yc_init = (py - cy_cam) * Z_init / fy
R_init = r_px * Z_init / fx

init_params = [Xc_init, Yc_init, R_init, Z_init]

# 最小二乘优化
res = least_squares(residuals, init_params, args=(pts,), method='lm')
Xc, Yc, R_mm, Z_mm = res.x

# 投影优化后的圆到像素坐标
proj_pts = project_circle([Xc, Yc, R_mm, Z_mm], n_points=360)

# 绘制结果
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 原始轮廓（绿色）
cv2.polylines(output, [pts.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=1)
# 粗拟合圆（蓝色）
cv2.circle(output, (int(px), int(py)), int(r_px), (255,0,0), 2)
# 优化后投影圆（红色）
cv2.polylines(output, [proj_pts.astype(np.int32)], isClosed=True, color=(0,0,255), thickness=2)
# 标注圆心和半径
cv2.putText(output, f"R={R_mm:.3f}mm, Z={Z_mm:.3f}mm", 
            (int(px+10), int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

# 保存结果
cv2.imwrite("measured_max_circle.png", output)

# 打印最大圆信息
print(f"最大圆像素半径: {r_px:.2f}px")
print(f"最大圆实际半径: {R_mm:.3f}mm")
print(f"最大圆物距 Z: {Z_mm:.3f}mm")
print(f"圆心像素坐标: ({px:.2f}, {py:.2f})")
