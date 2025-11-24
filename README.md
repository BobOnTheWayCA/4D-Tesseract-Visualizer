# 4D Tesseract Visualizer

A rotating 4D cube (tesseract) animated in Python with NumPy and Matplotlib.

> This demo relies on a graphical Matplotlib backend.  
> Run it on a desktop environment; a headless terminal alone will not display the window.

<p align="center">
  <img src="tesseract_demo.gif" alt="4D Tesseract rotation demo" width="1000">
</p>

---

## Repository Layout

- `tesseract.py`  
  Main script that builds the tesseract, applies 4D rotations, projects to 3D, and animates the result.

Key functions:

- `make_tesseract_vertices(side=3.0)`  
  Generates the 16 vertices of a 4D hypercube with coordinates in \([-side/2, +side/2]\).  
  Returns a `(4, 16)` NumPy array of \((x, y, z, w)\) coordinates.

- `make_tesseract_edges()`  
  Builds the edge list for the tesseract.  
  Two vertices are connected if their 4-bit indices differ in **exactly one bit** (Hamming distance 1).

- `rotation_4d(angle_xw, angle_yz)`  
  Returns a `4x4` rotation matrix that:
  - Rotates in the **x–w** plane by `angle_xw`
  - Rotates in the **y–z** plane by `angle_yz`

- `project_4d_to_3d(points_4d, focal=2.5)`  
  Applies a simple 4D → 3D perspective-like projection  
  - X = x / (focal − w)
  - Y = y / (focal − w)
  - Z = z / (focal − w)
  with `np.clip` to avoid division by zero when `focal ≈ w`.

- `init()`  
  Initializes all Matplotlib `Line3D` objects for the edges (empty lines before the animation starts).

- `update(frame)`  
  Per-frame callback for `FuncAnimation`:
  - Updates 4D rotation angles based on `frame`
  - Rotates all vertices in 4D
  - Projects them to 3D
  - Updates each edge with new coordinates, line width, and alpha for a pulsing, glowing effect.

---

## How to Run (macOS / Windows / Linux)

**Requirements**

- Python 3.9+  
- NumPy  
- Matplotlib (with a GUI backend)

**Install dependencies**

```bash
pip install numpy matplotlib
```

**Run script**

```bash
python3 tesseract.py
```

## License and Credits

- Individual work of **Shijie (Bob) Bu**
- University of Alberta, Department of Computing Science  


<div align="right">

<img src="UofAlbertalogo.svg" alt="University of Alberta Logo" width="330px" style="vertical-align: middle;">
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
Department of Computing Science
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
Spring 2025, Edmonton, AB, Canada
</p>


</div>

