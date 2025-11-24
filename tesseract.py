import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Tesseract (4D cube) vertices and edges
def make_tesseract_vertices(side=2.0):
    # Return (4,16) array of 4D coordinates for a tesseract of 'side' length
    # Coordinates range from -side/2 to +side/2 in x, y, z, w
    s = side / 2.0
    coords = []
    # 4D hypercube has 2^4=16 corners, each coordinate ±(side/2)
    for x in [-s, s]:
        for y in [-s, s]:
            for z in [-s, s]:
                for w in [-s, s]:
                    coords.append([x, y, z, w])
    return np.array(coords, dtype=np.float32).T  # shape (4, 16)

def make_tesseract_edges():
    # Return a list of (i, j) pairs, each indicating an edge between
    # vertices i and j in a tesseract. Two vertices share an edge if
    # they differ in exactly one coordinate bit (i.e., exactly one bit
    # of their 4-bit index is different)
    edges = []
    for i in range(16):
        # Get 4 bits of i
        ibits = [(i >> b) & 1 for b in range(4)]
        for j in range(i + 1, 16):
            jbits = [(j >> b) & 1 for b in range(4)]
            # Count bits that differ
            diff = sum(abs(ibits[b] - jbits[b]) for b in range(4))
            if diff == 1:
                edges.append((i, j))
    return edges

# 4D Rotataions and projection
def rotation_4d(angle_xw, angle_yz):
    # Return a (4x4) rotation matrix in 4D that:
    # rotates in the x–w plane by angle_xw
    # rotates in the y–z plane by angle_yz
    R = np.eye(4, dtype=np.float32)

    # x–w plane rotation (x <-> w are indices 0,3)
    cxw = np.cos(angle_xw)
    sxw = np.sin(angle_xw)
    R_xw = np.eye(4, dtype=np.float32)
    R_xw[0, 0] =  cxw
    R_xw[0, 3] = -sxw
    R_xw[3, 0] =  sxw
    R_xw[3, 3] =  cxw

    # y–z plane rotation (y <-> z are indices 1,2)
    cyz = np.cos(angle_yz)
    syz = np.sin(angle_yz)
    R_yz = np.eye(4, dtype=np.float32)
    R_yz[1, 1] =  cyz
    R_yz[1, 2] = -syz
    R_yz[2, 1] =  syz
    R_yz[2, 2] =  cyz

    return R_xw @ R_yz

def project_4d_to_3d(points_4d, focal=2.0):
    # A simple perspective-like projection from 4D to 3D
    # We'll treat w as if it influences distance
    # X = x / (focal - w)
    # Y = y / (focal - w)
    # Z = z / (focal - w)
    x = points_4d[0]
    y = points_4d[1]
    z = points_4d[2]
    w = points_4d[3]

    # Solved the case when w = focal, as this would lead to division by zero. 
    # In 3D perspective projection, we divide by z (distance from the camera), but z is always positive, 
    # so we don't worry about division by zero. However in 4D perspective projection, w can take any value, 
    # including values that make focal - w = 0, which would cause a mathematical error.
    # The 'focal' parameter enforces numerical stability by shifting the reference point for w, preventing extreme scaling 
    # when w approaches 0 or becomes negative. Without 'focal', division by small or negative w values could cause rapid, 
    # uncontrolled changes in projection, making the visualization unstable or flipping points inside out.
    denom = np.clip(focal - w, 0.01, None)  # Limit the minimum difference, as for preventing / 0
    X = x / denom
    Y = y / denom
    Z = z / denom
    return np.vstack((X, Y, Z))

# 3) Animation
# Set dark background
plt.style.use('dark_background')

# Prepare data
verts_4d = make_tesseract_vertices(side=3)  # (4,16)
edges = make_tesseract_edges()                # list of (i, j)
num_edges = len(edges)

# Create figure and 3D axis
fig = plt.figure(figsize=(8,8))
fig.patch.set_facecolor("black")
ax = fig.add_subplot(projection='3d')

# Remove the background grid and pane
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.grid(False)

# Make the axis background dark
ax.set_facecolor("black")
ax.set_box_aspect((1,1,1))
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_zlim(-4,4)
ax.grid(False)
ax.set_title("Rotating 4D Cube (Tesseract)", color='w')

# Store line objects for each edge
lines = []
cmap = plt.get_cmap('plasma')  # For intense colors
for idx, (i,j) in enumerate(edges):
    # Each edge gets an initial line with a unique color from the 'plasma' colormap
    color = cmap(float(idx) / (num_edges - 1))
    ln, = ax.plot([], [], [], color=color, lw=1.5, alpha=0.8)
    lines.append(ln)

def init():
    # Initialize the animation: empty lines.
    for ln in lines:
        ln.set_data_3d([], [], [])
    return lines

def update(frame):
    # At each frame:
    # 1) Compute 4D rotation angles
    # 2) Rotate all vertices
    # 3) Project to 3D
    # 4) Update each edge's 3D line, with dynamic thickness and alpha

    # Rotation speed
    angle_xw = 0.005 * frame
    angle_yz = 0.007 * frame

    # 4D rotation
    R_4d = rotation_4d(angle_xw, angle_yz)
    rotated_4d = R_4d @ verts_4d  # (4,16)

    # 4D->3D projection
    projected_3d = project_4d_to_3d(rotated_4d, focal=2.5)
    x3, y3, z3 = projected_3d[0], projected_3d[1], projected_3d[2]

    # For a dynamic line thickness and alpha, we use sin/cos (oscillation function)
    # Vary thickness and alpha slightly with 'frame' plus the edge index
    for e_idx, (ln, (i, j)) in enumerate(zip(lines, edges)):
        # Dynamic alpha
        alpha = 0.6 + 0.4 * np.sin(0.05 * frame + e_idx)
        alpha = np.clip(alpha, 0.2, 1.0)
        ln.set_alpha(alpha)

        # Dynamic line thickness
        lw = 1.5 + 2.5 * abs(np.sin(0.03 * frame + e_idx))
        ln.set_linewidth(lw)

        # Set new 3D coords for the edge
        ln.set_data_3d([x3[i], x3[j]],
                       [y3[i], y3[j]],
                       [z3[i], z3[j]])
    return lines

# Create animation
ani = FuncAnimation(
    fig, update, frames=range(1000),
    init_func=init, interval=25, blit=False, repeat=True
)

plt.show()
