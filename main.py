import math
import os
import threading
from math import floor, ceil
import time
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import beta
from scipy.stats import kde
from scipy.stats import rv_continuous
from scipy import integrate
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from scipy.stats import ss
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt 
import tkinter as tk
from tkinter import ttk
import random
import awkde
import numpy as np

inputFile = "COSYinput.txt"
dimensions = 4


def sample_gaussian():
    mnx=-3 # Lowest value of domain
    mx=3 # Highest value of domain
    mny=-3 # Lowest value of range
    my=3 # Highest value of range
    bound=1 # Upper bound of PDF value
    while True: # Do the following until a value is returned
       # Choose an X inside the desired sampling domain.
       x=random.uniform(mnx,mx)
       y=random.uniform(mny, my)
       # Choose a Y between 0 and the maximum PDF value.
       z=random.uniform(0,bound)
       # Calculate PDF
       pdf=(math.exp(- (((x)**2 + (y)**2) ) ))
       # Does (x,y) fall in the PDF?
       if z<pdf:
           # Yes, so return x
           return x,y
       # No, so loop


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, minx=np.NAN, miny=np.NAN, maxx=np.NAN, maxy=np.NAN, **kwargs): 
    if (minx==np.NAN):
        minx = int(float(min(x)))
    if (miny==np.NAN):
        miny = int(float(min(y)))
    if (maxx==np.NAN):
        maxx = int(float(max(x)))
    if (maxy==np.NAN):
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

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def getSamples(useCOSYInput = False, numSamples = 1000, x=-1, y=-1):   
    if (useCOSYInput):
        times = []
        radiuses = []
        f = open(inputFile, 'r')
        lines = f.readlines()
        for i in range (len(lines)):
            if (i == 0):
                continue
            data = lines[i].split(" ")
            if x == -1 or y == -1:
                x = 0
                y = 0
            times.append(float(data[x]))
            radiuses.append(float(data[y]))
        return times, radiuses
    
    yrand = []
    xrand = []
    for i in range (numSamples):
       x, y = sample_gaussian()
       yrand.append(y)
       xrand.append(x)
    return xrand, yrand


def plot(xx, yy, zz, x, y, axis, pltX, pltY):
    a = axis[pltX, pltY]
    #plt.axis('off')
    a.set_xlim([min(x), max(x)])
    a.set_ylim([min(y), max(y)])

    axis[pltX, pltY].pcolormesh(xx, yy, zz)
    axis[pltX, pltY].scatter(x, y, s=2, facecolor='gray')

def generate(useCOSYInput, bandwidth=0.01, resolution=100j, numSamples = 1000):
    fig, axis = plt.subplots(dimensions, dimensions)
    for x in range (dimensions):
        for y in range (dimensions):
            samples = getSamples(useCOSYInput, numSamples, x, y)
            xSamples = samples[0]
            ySamples = samples[1]
            xx, yy, zz = kde2D(xSamples, ySamples, bandwidth, xbins=resolution, ybins=resolution, minx=floor(float(min(xSamples))), maxx=ceil(float(max(xSamples))), miny=floor(float(min(ySamples))), maxy=ceil(float(max(ySamples))))
            plot(xx, yy, zz, xSamples, ySamples, axis, x, y)
            
            averageX, stdX = weighted_avg_and_std(xx, zz)
            averageY, stdY = weighted_avg_and_std(yy, zz)
            #print("Standard Deviation: " + str(np.std(zz)))
            print("Average on X: " + str(averageX))
            print("Average on Y: " + str(averageY))
            print("Average: " + str(math.sqrt( averageX**2 + averageY ** 2 )))

            print("Standard Deviation on X: " + str(stdX))
            print("Standard Deviation on Y: " + str(stdY))
            print("Standard Deviation: " + str(math.sqrt( stdX**2 + stdY ** 2 )))

    
    plt.show()


def on_enter_button_click():
    # Placeholder function for button click event
    # You can add functionality to process the entered function here
    print("Enter button clicked")
# Global variables
fig = None
ax = None
canvas = None

def compute_gaussian():
    yScatter = []
    xScatter = []
    for i in range (2000):
       x, y = sample_gaussian()
       yScatter.append(y)
       xScatter.append(x)
    bandwidth = 1
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution, minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))), miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))
    return xScatter, yScatter, xx, yy, zz

def plot_gaussian():
    """Plot Gaussian distribution."""
    global ax, canvas
    thread = threading.Thread(target=compute_gaussian)
    yScatter = [1,1]
    xScatter = [1,1]
    xx = [1,1,1]
    yy=[1,1,1]
    zz = [1,1,1]
    thread.start()

    if (not thread.is_alive()):
        print("done")
   # for i in range (2000):
   #    x, y = sample_gaussian()
   #    yScatter.append(y)
   #   xScatter.append(x)
    #bandwidth = 1
    #resolution = 400j

   #xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution, minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))), miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))
    

    ax.pcolormesh(xx, yy, zz)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')

    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Gaussian Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    #getSamples()
    
    canvas.draw()

def plot_linear():
    """Plot linear function."""
    global ax, canvas
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    
    ax.clear()
    ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    ax.set_title('Linear Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    canvas.draw()

def plot_gaussian_modified():
    """Plot modified Gaussian function."""
    global ax, canvas
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2 - Y**2 + X*Y)
    
    ax.clear()
    ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    ax.set_title('Gaussian Modified Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    canvas.draw()

def plot_sine():
    """Plot sine function."""
    global ax, canvas
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.sin(Y)
    
    ax.clear()
    ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

    ax.set_title('Sine Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    canvas.draw()

def create_plot_frame(parent):
    """Create and return the plot frame with buttons."""
    frame = ttk.Frame(parent)
    frame.pack(pady=10, padx=10, fill='both', expand=True)

    # Create buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack()

    ttk.Button(button_frame, text="Gaussian", command=plot_gaussian).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Linear", command=plot_linear).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Gaussian Modified", command=plot_gaussian_modified).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Sine", command=plot_sine).pack(side='left', padx=5)

    # Create plot area
    global fig, ax, canvas
    fig = Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)

    return frame

def create_data_plot_tab(parent):
    """Create and return the data plot tab frame."""
    frame = ttk.Frame(parent)
    label = ttk.Label(frame, text="This is the Data Plot tab")
    label.pack(pady=20)
    return frame

def main():
    """Main function to set up the GUI application."""
    global fig, ax, canvas, root

    root = tk.Tk()
    root.title("Windows 7 Style Window")
    root.geometry("1280x720")

    # Create and configure notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Create and add tabs
    function_plot_tab = create_plot_frame(notebook)
    data_plot_tab = create_data_plot_tab(notebook)

    notebook.add(function_plot_tab, text="Function Plot")
    notebook.add(data_plot_tab, text="Data Plot")

    # Set default plot
    #plot_gaussian()

    root.mainloop()

if __name__ == "__main__":
    main()