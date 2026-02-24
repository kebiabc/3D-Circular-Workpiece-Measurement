# 3D Circular Workpiece Measurement (Stereo Vision)

Reconstruct a circular workpiece in 3D and estimate its real-world diameter from a calibrated stereo pair.
This repo demonstrates a full pipeline: **circle detection → stereo triangulation → metric diameter estimation → 3D visualization**.

---
<img width="1000" height="800" alt="Figure_1" src="https://github.com/user-attachments/assets/45d9a41b-68c0-44d3-b70e-1c9b7935dae0" />

## Features

* Robust circle edge extraction (Gaussian blur + Canny + contour search)
* Sub-pixel refinement of edge points
* Best-fit enclosing circle per view
* Stereo **triangulation** of the circle center using projection matrices
* Metric **diameter** estimation from pixel radius + depth (pinhole camera model)
* Matplotlib 3D visualization of the reconstructed circle and cameras

---

## Requirements

* Python 3.8+
* OpenCV (`opencv-python`)
* NumPy
* Matplotlib

Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` (example)

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
│   ├── left.jpg          # left view image
│   └── right.jpg         # right view image
├── measure_stereo.py        # the code you pasted
└── README.md
```

**Outputs**

* `tools/left_result.jpg`, `tools/right_result.jpg` – detection overlays
* `3d_reconstruction.png` – 3D plot (saved when visualization runs)
* Console logs: circle centers (px), radii (px), 3D center (mm), diameter (mm), depth (mm)

---

## Usage

1. Put your stereo images into `tools/left.jpg` and `tools/right.jpg`.
2. Edit camera parameters in `main()`:

   * `K1`, `K2` (intrinsics)
   * `R`, `T` (right camera pose w\.r.t. left)
   * (Optional) Distortion coefficients if you plan to undistort beforehand
3. Run:

```bash
python measure_stereo.py
```

---

## How It Works (Pipeline)

1. **2D Circle Fitting – `fit_circle(image_path)`**

   * Grayscale → Gaussian blur → Canny edges
   * `findContours` to collect edges
   * For each contour, compute `minEnclosingCircle`; keep the one with the **largest radius**
   * Sub-pixel refine contour points (`cornerSubPix`), recompute enclosing circle
   * Return **center (u, v)** and **radius r\_px** in pixels

## Stereo Triangulation – `triangulate_points(point1, point2, P1, P2)`

Build camera projection matrices:

$$
P_1 = K_1 [I \mid 0], \quad
P_2 = K_2 [R \mid T]
$$

Use `cv2.triangulatePoints` on the matched **circle centers** from the two views.

Convert homogeneous 4D back to 3D (mm) → **circle center in 3D**

---

## Metric Diameter – `calculate_diameter(...)`

With depth $Z$ of the 3D center in each camera, convert pixel radius to metric:

$$
r_{\text{real}} =
\frac{r_{\text{px}} \cdot Z}{f_x}
$$

Compute two diameters (one per view) and **average** them.

---

## Visualization – `visualize_3d_points(...)`

Plot:

- 3D center
- a circle lying in a plane at the recovered depth
- camera markers

---

## Math Notes

### Pinhole projection

$$
\begin{aligned}
u &= \frac{f_x X}{Z} + c_x \\\\
v &= \frac{f_y Y}{Z} + c_y
\end{aligned}
$$

---

### Pixel radius → metric diameter

$$
\text{diameter}
=
2 \, r_{\text{real}}
=
2 \,
\frac{r_{\text{px}} \cdot Z}{f_x}
$$

---

### Stereo geometry

Given:

$$
(u_1, v_1),
(u_2, v_2)
$$

and projection matrices:

$$
P_1, P_2
$$

Triangulation finds:

$$
\mathbf{X}
=
(X, Y, Z)^T
$$
---

## Configuration Tips

* **Edge thresholds** (`Canny(50, 150)`) may need tuning for your lighting and texture.
* **Sub-pixel refinement** window size `(5,5)` and termination criteria can be adjusted for stability.
* If multiple circular objects exist, consider filtering contours by **area**, **circularity**, or expected **radius range**.

---

## Limitations

* Assumes the visible contour is close to a **true circle projection** (thin circular rim).
  Strong perspective or partial occlusion may bias the enclosing circle.
* Pixel radius uses a **single depth (circle center)**; for very thick objects or tilted faces,
  a full **conic fit** (ellipse in image) and plane estimation would be more accurate.
* No explicit **outlier rejection** (e.g., RANSAC) on edge points; you can add it for robustness.

---

## Troubleshooting

* **“Cannot read image”**: Check file paths (`tools/left.jpg`, `tools/right.jpg`).
* **“No valid contour found”**: Loosen Canny thresholds, improve lighting, or blur less.
* **Unrealistic diameter**:

  * Verify calibration scales (are `T` and desired output both in **mm**?)
  * Make sure images are **undistorted**.
  * Check that left/right centers correspond to the **same physical circle**.
* **Triangulation looks wrong**:

  * Confirm `R`, `T` direction (right **from** left), and consistent coordinate system.
  * Ensure `P1 = K1 [I|0]`, `P2 = K2 [R|T]` with the **same units**.

---

## Extensions (Ideas)

* Undistort/rectify before detection, or detect on rectified images and use **disparity** for depth sanity checks.
* Replace enclosing circle with **ellipse fit** to handle oblique views and estimate plane pose.
* RANSAC on edge points to reject outliers before fitting.
* Bundle adjustment: jointly refine 3D center and diameter by minimizing **reprojection error** in both views.

---

## License

Add your preferred license here (e.g., MIT).

---

## Acknowledgements

* OpenCV for vision primitives
* Matplotlib for visualization

