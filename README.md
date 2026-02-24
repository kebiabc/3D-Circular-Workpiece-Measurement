# 3D Circular Workpiece Measurement (Stereo Vision)

Reconstruct a circular workpiece in 3D and estimate its real-world diameter from a calibrated stereo pair.  
This repo demonstrates a full pipeline:

**circle detection → stereo triangulation → metric diameter estimation → 3D visualization**

---

<img width="1000" height="800" alt="Figure_1" src="https://github.com/user-attachments/assets/45d9a41b-68c0-44d3-b70e-1c9b7935dae0" />

---

## Features

- Robust circle edge extraction (Gaussian blur + Canny + contour search)
- Sub-pixel refinement of edge points
- Best-fit enclosing circle per view
- Stereo triangulation of the circle center using projection matrices
- Metric diameter estimation from pixel radius and depth (pinhole model)
- Matplotlib 3D visualization of reconstructed geometry

---

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install:

```bash
pip install -r requirements.txt
```

requirements.txt

```
opencv-python>=4.7
numpy>=1.23
matplotlib>=3.6
```

---

## File Structure

```
.
├── tools/
│   ├── left.jpg
│   └── right.jpg
├── measure_stereo.py
└── README.md
```

Outputs:

- tools/left_result.jpg  
- tools/right_result.jpg  
- 3d_reconstruction.png  

Console output includes:

- circle center (px)
- radius (px)
- 3D center (mm)
- diameter (mm)
- depth (mm)

---

## Usage

1. Put stereo images:

```
tools/left.jpg
tools/right.jpg
```

2. Edit camera parameters in main():

- K1, K2 (intrinsics)
- R, T (extrinsics)

3. Run:

```bash
python measure_stereo.py
```

---

## Pipeline

### 1. 2D Circle Fitting

Function:

```
fit_circle(image_path)
```

Steps:

- grayscale
- Gaussian blur
- Canny edge detection
- findContours
- minEnclosingCircle
- cornerSubPix refinement

Returns:

```
center (u, v)
radius r_px
```

---

### 2. Stereo Triangulation

Projection matrices:

$$
P_1 = K_1 \begin{bmatrix} I & 0 \end{bmatrix}
$$

$$
P_2 = K_2 \begin{bmatrix} R & T \end{bmatrix}
$$

Triangulation:

$$
\mathbf{X} =
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
$$

Computed using:

```
cv2.triangulatePoints
```

---

### 3. Metric Diameter Estimation

Pixel radius to metric radius:

$$
r_{real}
=
\frac{r_{px} \cdot Z}{f_x}
$$

Diameter:

$$
diameter =
2 \cdot r_{real}
=
2 \cdot \frac{r_{px} \cdot Z}{f_x}
$$

Average diameter from both cameras.

---

### 4. Visualization

Plot:

- 3D circle center
- reconstructed circle
- camera locations

---

## Camera Model

Pinhole projection:

$$
u =
\frac{f_x X}{Z}
+
c_x
$$

$$
v =
\frac{f_y Y}{Z}
+
c_y
$$

---

## Stereo Geometry

Given image points:

$$
(u_1, v_1)
$$

$$
(u_2, v_2)
$$

Triangulation computes:

$$
(X, Y, Z)
$$

---

## Calibration Notes

Use accurate calibration:

Intrinsics:

$$
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

Extrinsics:

$$
R, T
$$

Units of T determine output units.

If T is in mm, output is in mm.

---

## Image Preparation

Undistort images before detection:

```
cv2.undistort
```

This improves accuracy.

---

## Configuration Tips

Adjust:

```
Canny(50,150)
```

Refinement window:

```
(5,5)
```

Optional filtering:

- area
- circularity
- expected radius

---

## Limitations

Assumes visible contour is circular.

Oblique viewing introduces ellipse projection error.

For higher accuracy use ellipse fitting and plane reconstruction.

---

## Troubleshooting

Cannot read image:

Check paths.

No contour found:

Adjust Canny thresholds.

Wrong diameter:

Check calibration units.

Wrong triangulation:

Verify:

$$
P_1 = K_1 [I \; 0]
$$

$$
P_2 = K_2 [R \; T]
$$

---

## Extensions

Possible improvements:

- stereo rectification
- ellipse fitting
- RANSAC circle fitting
- bundle adjustment refinement

---

## License

MIT (or your preferred license)

---

## Acknowledgements

OpenCV  
Matplotlib
