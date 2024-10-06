import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from math import floor, ceil

# 2D Gaussian Kernel Definition
def gaussian_kernel_2d(x, y, xi, yi, bandwidthX, bandwidthY):

    normalization = 1 / (2 * np.pi * bandwidthX * bandwidthY)
    exponent = -0.5 * ( ((x - xi) **2) / (bandwidthX**2) + ((y - yi)**2) / (bandwidthY**2) )
    return normalization * np.exp(exponent)

# 2D Kernel Density Estimation Function
def kde_2d(x, y, xData, yData, bandwidthX, bandwidthY):

    return np.sum([gaussian_kernel_2d(x, y, xi, yi, bandwidthX, bandwidthY) for xi, yi in zip(xData, yData)]) / len(xData)

# Function to compute KDE for a grid of points
def custom_kde2d(x_grid, y_grid, xData, yData, bandwidthX, bandwidthY):

    z = np.zeros_like(x_grid)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            z[i, j] = kde_2d(x_grid[i, j], y_grid[i, j], xData, yData, bandwidthX, bandwidthY)
    return z


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, minx=np.nan, miny=np.nan, maxx=np.nan, maxy=np.nan, **kwargs): 
    if (minx==np.nan):
        minx = int(float(min(x)))
    if (miny==np.nan):
        miny = int(float(min(y)))
    if (maxx==np.nan):
        maxx = int(float(max(x)))
    if (maxy==np.nan):
        maxy = int(float(max(y)))
        
    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[minx:maxx:xbins, 
                      miny:maxy:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)
    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

def kde2D_b(xVals, yVals, xBand, yBand, xbins=100j, ybins=100j): 
    # Do the Kernel Density 
    return