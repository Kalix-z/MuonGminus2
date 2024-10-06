import numpy as np
import matplotlib.pyplot as plt
import muongminus2calculations
from kde2d import gaussian_kernel_2d

# Define the range and resolution of x and y
x_vals = np.linspace(-10, 10, 1000)  # 100 points from -3 to 3
y_vals = np.linspace(-10, 10, 1000)  # 100 points from -3 to 3

# Create a meshgrid for x and y
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

# Compute the function e^(-x^2 - y^2)
z_values = gaussian_kernel_2d(x_grid, y_grid, 1, 0, 2.5, 2)

# Create the pcolormesh plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(x_grid, y_grid, z_values, shading='auto')  # shading='auto' for smoother color transition
plt.colorbar(label='e^(-x^2 - y^2)')
plt.title(r'$e^{-x^2 - y^2}$')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')  # Keep the aspect ratio of the plot equal
plt.show()
