import sample_distributions as sd
from kde2d import kde2D
from math import floor, ceil
import matplotlib.pyplot as plt


def plot_gaussian(numPoints, bw):
    """Plot Gaussian distribution with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    # Sample Gaussian points
    for i in range(numPoints):
        x, y = sd.sample_gaussian()
        yScatter.append(y)
        xScatter.append(x)
    
    bandwidth =  bw  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xx, yy, zz)   
    fig.colorbar(c, ax=ax)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_title('Gaussian Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()



def plot_linear(numPoints, bw):
    """Plot linear function with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    for i in range(numPoints):
        x, y = sd.sample_linear()
        yScatter.append(y)
        xScatter.append(x)

    bandwidth = bw  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))
    fig, ax = plt.subplots()
    ax.clear()
    c = ax.pcolormesh(xx, yy, zz)
    fig.colorbar(c, ax=ax)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Linear Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.plot()


def plot_gaussian_modified(numPoints, bw):
    """Plot modified Gaussian function with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    for i in range(numPoints):
        x, y = sd.sample_gaussian_modified()
        yScatter.append(y)
        xScatter.append(x)

    bandwidth = bw  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))

    fig, ax = plt.subplots() 
    ax.clear()
    c = ax.pcolormesh(xx, yy, zz)
    fig.colorbar(c, ax=ax)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Modified Gaussian Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.plot()


def plot_sine(numPoints, bw):
    """Plot sine function with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    for i in range(numPoints):
        x, y = sd.sample_sine()
        yScatter.append(y)
        xScatter.append(x)

    bandwidth = bw  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))

    fig, ax = plt.subplots()
    
    ax.clear()
    c = ax.pcolormesh(xx, yy, zz)
    fig.colorbar(c, ax=ax)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Sine Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.plot()

