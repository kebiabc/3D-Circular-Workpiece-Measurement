import cv2
import numpy as np

# 读取灰度图
img = cv2.imread("tools/6.jpg", cv2.IMREAD_GRAYSCALE)

# 高斯模糊去噪
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Canny 边缘检测
edges = cv2.Canny(blur, 50, 150)

# 轮廓提取（用 RETR_TREE 保留内外轮廓关系）
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 转成彩色图
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i, cnt in enumerate(contours):
    # 拟合圆
    if len(cnt) >= 5:  # 点数太少无法拟合
        pts = np.array(cnt, dtype=np.float32).reshape(-1, 2)
        center, radius = cv2.minEnclosingCircle(pts)
        
        # 绘制圆
        cv2.circle(output, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
        cv2.circle(output, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
        
        # 标记编号
        cv2.putText(output, f"{i}", (int(center[0]) - 10, int(center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 保存结果
cv2.imwrite("all_circles_fit.png", output)
print(f"总共检测到 {len(contours)} 个轮廓")
