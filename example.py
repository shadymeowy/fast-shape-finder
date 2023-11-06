from fast_shape_finder import simple_edge_hsv, ransac_ellipse, simple_denoise, calculate_lch_bounds
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import cv2

# HSV constraints for reddish colors
hue_limit = np.deg2rad(-30), np.deg2rad(+30)
chroma_limit = 0.3, 1.
intensity_limit = 0.2, 1.
bounds = calculate_lch_bounds(hue_limit, chroma_limit, intensity_limit)

def plot_ellipse(x, y, a, b, theta):
    o = np.linspace(0, 2 * np.pi)
    plt.plot(
        x + a * np.cos(theta) * np.cos(o) - b * np.sin(theta) * np.sin(o),
        -(y + a * np.sin(theta) * np.cos(o) + b * np.cos(theta) * np.sin(o))
    )


def draw_ellipse(image, x, y, a, b, theta):
    return cv2.ellipse(
        image.copy(),
        [int(x), int(y)],
        [int(a), int(b)],
        180 * theta / pi,
        0,
        360,
        [0, 255, 0],
        2
    )


# Load the image
image = np.array(cv2.imread("image.png")[:, :, ::-1])
h = image.shape[0]
w = image.shape[1]

# Delta parameter for the RANSAC algorithm
delta = 2

# Initialize two buffers for the edge points
pps = np.zeros((2048, 2), dtype="int32")
pps_d = np.zeros((2048, 2), dtype="int32")

# Extract the edge points
ln = simple_edge_hsv(image, pps, bounds._asdict(), 256, 256)
print("Number of edge points", ln)

# Apply denoising to the edge points
ln_d = simple_denoise(pps[:ln], pps_d, 30)
print("Number of edge points after denoising", ln_d)

# Check the number of edge points
if ln_d > 10:
    # Run the RANSAC algorithm
    e = ransac_ellipse(pps_d[:ln_d], 5000, 25, 150, 10, delta)
else:
    # Raise an error if the number of edge points is too small
    raise Exception("No ellipse found")

# Plot both the original and the denoised edge points
plt.scatter([p[0] for p in pps[:ln]], [-p[1] for p in pps[:ln]])
plt.scatter([p[0] for p in pps_d[:ln_d]], [-p[1] for p in pps_d[:ln_d]])
# Plot the ellipse
plot_ellipse(**e)
plt.gca().set_aspect('equal')
plt.show()

# Draw the ellipse on the image
image = draw_ellipse(image.copy(), **e)
plt.imshow(image)
plt.show()