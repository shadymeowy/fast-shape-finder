# fast-shape-finder
Finding simple geometric shapes quickly using RANSAC-based algorithms

This is an attempt to apply RANSAC to find geometric shapes in images. The ultimate goal of this project is to find arbitrary shapes on any surface and to acquiring
both the position and attitude information.

In the first step, the edge features are extracted from the image by applying inequalities derived from HSV to RGB conversion formulas.
This step is equivalent to converting images to HSV space, thresholding the individual pixels, and applying edge detection.
However, this approach is generally quite slow since it involves extensive use of floating-point arithmetic operations like division.
In my approach, all of them are applied in a single pass on the image. It also does not involve any floating-point arithmetic or even integer division.
The "edge detection" method used is quite primitive. It is simple thresholding across the horizontal or
vertical axis and recording the color change points.
Yet, it works pretty reliably, given that the HSV filter is working reasonably. In this way, cache misses are minimized. It should be noted that this step does not
scan the complete image. It uses Halton sequence to select rows or columns in a Monte Carlo manner.

In the second step, a simple "noise reduction" algorithm is applied to acquired edge points. This step is optional, but it generally significantly improves the third step's efficiency. It checks the pixel-wise distance between edge points in the same row or column. If some portions
are close to each other, this step merges them. If some portions are small enough, it eliminates them.

In the third step, the RANSAC algorithm is applied to fit shapes. While only circles and ellipses are allowed at this point, the plan is
to use vertices and edges to acquire the 2D perspective transformation of the shape. In this way, it will be possible to find arbitrary shapes 
under perspective transformation. Later, that transformation will be combined with the known properties of the used camera, which will result in
the exact relative position and attitude.

Ellipse fit utilizes the general conic equation and a custom linear equation solver. Then, some checks are performed to decide whether or not
the found conic equation corresponds to an ellipse.

The code is optimized for branch and cache misses. It takes only ~500 microseconds to find a red ellipse in a 1920x1080 photo, which includes
an extensive amount of noise with the same color on Ryzen 4800H processor.

An alternative method is offered for the first step, which does not use HSV inequalities at all. Instead, it applies an inequality check based on a color
distancing formula. Again, this method does not perform any floating-point arithmetic.
