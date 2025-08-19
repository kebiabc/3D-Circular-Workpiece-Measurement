import cv2
import numpy as np

# === 内参矩阵 ===
fx = 6350.955187
fy = 6352.45175107693
cx = 2527.69608602686
cy = 2545.5974390674
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# === 畸变系数 ===
distCoeffs = np.array([-0.063281104, 0.079497179, 
                       -0.000429296, -0.000355795, 
                        0.061829162], dtype=np.float64)

# === 读取原图 ===
img = cv2.imread("tools/1.jpg")

# === 去畸变 ===
h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
dst = cv2.undistort(img, K, distCoeffs, None, new_K)

# 保存或显示
cv2.imwrite("undistorted.jpg", dst)
