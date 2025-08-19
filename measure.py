import cv2
import numpy as np
import math

# 读取灰度图
img = cv2.imread("tools/right.jpg", cv2.IMREAD_GRAYSCALE)

# 高斯模糊去噪
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Canny 边缘检测
edges = cv2.Canny(blur, 50, 150)

# 轮廓提取
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print(contours)
# 轮廓提取（用 RETR_TREE 保留内外轮廓关系）
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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

# 用最小二乘法拟合圆（亚像素）
# 把轮廓点转成 float
# 亚像素优化
pts = np.array(best_contour, dtype=np.float32).reshape(-1, 1, 2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
cv2.cornerSubPix(img, pts, (5,5), (-1,-1), criteria)
pts = pts.reshape(-1, 2)
center, radius = cv2.minEnclosingCircle(pts)

# 在原图上画结果
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.circle(output, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
cv2.circle(output, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)

# 保存结果
cv2.imwrite("outer_circle_fit.png", output)

print(center, radius)
