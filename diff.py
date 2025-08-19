import cv2
import numpy as np

# 读取与预处理（与上一步一致）
img = cv2.imread("tools/undistorted.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 提取最大外轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# max_contour = max(contours, key=cv2.contourArea)
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

# ---------- OpenCV最小外接圆（参考） ----------
center_a, radius_a = cv2.minEnclosingCircle(pts.astype(np.float32))

# ---------- 最小二乘圆拟合（Kåsa/代数法） ----------
# 代数形式：x^2 + y^2 + ax + by + c = 0
x = pts[:, 0]
y = pts[:, 1]
A = np.column_stack([x, y, np.ones_like(x)])
b = -(x**2 + y**2)
# 最小二乘解
coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
a, b_coef, c = coeff
cx = -a / 2.0
cy = -b_coef / 2.0
R = np.sqrt(cx**2 + cy**2 - c)

# ---------- 残差统计（几何残差：点到圆的半径偏差） ----------
ri = np.sqrt((x - cx)**2 + (y - cy)**2)
residual = ri - R  # +为外侧，-为内侧
rmse = np.sqrt(np.mean(residual**2))
mae = np.mean(np.abs(residual))
max_abs = np.max(np.abs(residual))
std = np.std(residual)

# 保存可视化：残差上色
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for (px, py, r) in zip(x.astype(int), y.astype(int), residual):
    # 将残差线性映射到颜色（蓝-0-红），仅做直观展示
    # clip在 [-3, 3] 像素范围
    rr = np.clip((r + 3) / 6, 0, 1)
    color = (int(255 * (1 - rr)), 0, int(255 * rr))  # B-R 过渡
    cv2.circle(vis, (px, py), 1, color, -1)

# 画拟合圆
cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(R)), (0, 255, 0), 2)
cv2.circle(vis, (int(round(cx)), int(round(cy))), 2, (0, 255, 255), -1)

cv2.imwrite("circle_fit_residual_vis.png", vis)

print(cx, cy, R, rmse, mae, std, max_abs, center_a, radius_a)
