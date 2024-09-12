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
from functools import partial
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
       pdf=math.exp(-(x**2)-(y**2))
       # Does (x,y) fall in the PDF?
       if z<pdf:
           # Yes, so return x
           return x,y
       # No, so loop

def sample_gaussian_modified():
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
       pdf=math.exp(-(x**2)-(y**2)+x*y)
       # Does (x,y) fall in the PDF?
       if z<pdf:
           # Yes, so return x
           return x,y
       # No, so loop



def sample_linear():
    mnx=-10 # Lowest value of domain
    mx=10 # Highest value of domain
    mny=-10 # Lowest value of range
    my=10 # Highest value of range
    bound=20 # Upper bound of PDF value
    min=-20
    while True: # Do the following until a value is returned
       # Choose an X inside the desired sampling domain.
       x=random.uniform(mnx,mx)
       y=random.uniform(mny, my)
       # Choose a Y between 0 and the maximum PDF value.
       z=random.uniform(min,bound)
       # Calculate PDF
       pdf=(x+y)
       # Does (x,y) fall in the PDF?
       if z<pdf:
           # Yes, so return x
           return x,y
       # No, so loop

def sample_sine():
    mnx=-5 # Lowest value of domain
    mx=5 # Highest value of domain
    mny=-5 # Lowest value of range
    my=5 # Highest value of range
    bound=2 # Upper bound of PDF value
    min=0
    while True: # Do the following until a value is returned
       # Choose an X inside the desired sampling domain.
       x=random.uniform(mnx,mx)
       y=random.uniform(mny, my)
       # Choose a Y between 0 and the maximum PDF value.
       z=random.uniform(min,bound)
       # Calculate PDF
       pdf=(np.sin(x*0.5)**2 + np.sin(y*0.5)**2)
       # Does (x,y) fall in the PDF?
       if z<pdf:
           # Yes, so return x
           return x,y
       # No, so loop


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

def getSamples(x=-1, y=-1):   
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
            samples = [0,0]#getSamples(useCOSYInput, numSamples, x, y)
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

def get_bandwidth():
    """Get the bandwidth value from the entry widget."""
    try:
        # Fetch the bandwidth entered by the user
        bandwidth = float(bandwidth_entry.get())
        return bandwidth
    except ValueError:
        # Default to 1 if the input is invalid
        print("Invalid bandwidth value, defaulting to 1.")
        return 1

def plot_gaussian():
    """Plot Gaussian distribution with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    # Sample Gaussian points
    for i in range(2000):
        x, y = sample_gaussian()
        yScatter.append(y)
        xScatter.append(x)
    
    bandwidth = get_bandwidth()  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))

    ax.clear()
    ax.pcolormesh(xx, yy, zz)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Gaussian Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    canvas.draw()


def plot_linear():
    """Plot linear function with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    for i in range(2500):
        x, y = sample_linear()
        yScatter.append(y)
        xScatter.append(x)

    bandwidth = get_bandwidth()  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))

    ax.clear()
    ax.pcolormesh(xx, yy, zz)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Linear Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    canvas.draw()


def plot_gaussian_modified():
    """Plot modified Gaussian function with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    for i in range(3000):
        x, y = sample_gaussian_modified()
        yScatter.append(y)
        xScatter.append(x)

    bandwidth = get_bandwidth()  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))

    ax.clear()
    ax.pcolormesh(xx, yy, zz)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
    ax.set_title('Modified Gaussian Function')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    canvas.draw()


def plot_sine():
    """Plot sine function with user-defined bandwidth."""
    global ax, canvas
    yScatter = []
    xScatter = []

    for i in range(2000):
        x, y = sample_sine()
        yScatter.append(y)
        xScatter.append(x)

    bandwidth = get_bandwidth()  # Get user-defined bandwidth
    resolution = 400j

    xx, yy, zz = kde2D(xScatter, yScatter, bandwidth, xbins=resolution, ybins=resolution,
                       minx=floor(float(min(xScatter))), maxx=ceil(float(max(xScatter))),
                       miny=floor(float(min(yScatter))), maxy=ceil(float(max(yScatter))))

    ax.clear()
    ax.pcolormesh(xx, yy, zz)
    ax.scatter(xScatter, yScatter, s=2, facecolor='gray')
    ax.set_xlim([min(xScatter), max(xScatter)])
    ax.set_ylim([min(yScatter), max(yScatter)])
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

    # Create bandwidth input
    global bandwidth_entry
    bandwidth_label = ttk.Label(button_frame, text="Bandwidth:")
    bandwidth_label.pack(side='left', padx=5)

    bandwidth_entry = ttk.Entry(button_frame)
    bandwidth_entry.pack(side='left', padx=5)
    bandwidth_entry.insert(0, "1")  # Default value for bandwidth

    # Create plot area
    global fig, ax, canvas
    fig = Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)

    return frame


def calculate_and_plot(bandwidth):
    """Perform calculation and plot the results."""
    global canvas, frame
    
    # Get the bandwidth from the entry
    resolution = 100j  # Set resolution or allow it to be a parameter
    dataName = ["", "time (ns)", "r (mm)", "y (mm)", "MoM (Gev)"]
    # Iterate through the 5x5 grid
    for i in range(1, 5):
        for j in range(1, 5):
            xVals, yVals = getSamples(i-1, j-1)
            if xVals and yVals:  # Ensure there are samples
                bandwidth_multiplier = 1#(float(max(xVals)) - float(min(xVals)))* (float(max(yVals)) - float(min(yVals)))
                xx, yy, zz = kde2D(xVals, yVals, float(bandwidth)*bandwidth_multiplier, xbins=resolution, ybins=resolution,
                                   minx=floor(float(min(xVals))), maxx=ceil(float(max(xVals))),
                                   miny=floor(float(min(yVals))), maxy=ceil(float(max(yVals))))
                
                # Create a Matplotlib figure with smaller size
                fig, ax = plt.subplots(figsize=(2, 2))  # Adjust the figsize to fit the grid
                ax.pcolormesh(xx, yy, zz)
                ax.scatter(xVals, yVals, s=2, color='gray')
                ax.set_xlim([min(xVals), max(xVals)])
                ax.set_ylim([min(yVals), max(yVals)])
                averageX, stdX = weighted_avg_and_std(xx, zz)
                averageY, stdY = weighted_avg_and_std(yy, zz)
                print(dataName[i-1] + " " + dataName[j-1] + ": ")
                print("average: ")
                print(math.sqrt(averageX**2+averageY**2))
                print("mean: ")
                print(math.sqrt(stdX**2+stdY**2))
                
                # Embed the plot in the Tkinter window
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=i, column=j, padx=5, pady=5, sticky='nsew')

def create_data_plot_tab(parent):
    global frame
    frame = ttk.Frame(parent)
    frame.pack(pady=10, padx=10, fill='both', expand=True)
    
    # Add a label and entry for bandwidth input
    bw_label = ttk.Label(frame, text="Bandwidth:")
    bw_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    
    bw_var = tk.StringVar()
    bw_entry = ttk.Entry(frame, textvariable=bw_var)
    bw_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    bw_entry.insert(0, "1")  # Default value for bandwidth
    
    # Button to perform calculation and plot
    calculate_button = ttk.Button(frame, text="Calculate and Plot", command=partial(calculate_and_plot, bw_var.get()))
    calculate_button.grid(row=0, column=2, padx=5, pady=5)

    # Adjust the grid configuration to account for the new row
    for i in range(6):
        frame.grid_columnconfigure(i, weight=1)
        if i == 0:
            frame.grid_rowconfigure(i, weight=0)  # First row with smaller weight
        else:
            frame.grid_rowconfigure(i, weight=1)
    
    dataName = ["", "time (ns)", "r (mm)", "y (mm)", "MoM (Gev)"]

    # Create a 5x5 grid starting from the second row
    for i in range(1, 5):
        for j in range(1, 5):
            if i == 1:
                # First row with text
                label = ttk.Label(frame, text=dataName[j])
                label.grid(row=i, column=j, padx=5, pady=5, sticky='nsew')
            elif j == 1:
                # First column with text
                label = ttk.Label(frame, text=dataName[i])
                label.grid(row=i, column=j, padx=5, pady=5, sticky='nsew')
    
    return frame

def main():
    """Main function to set up the GUI application."""
    global fig, ax, canvas, root

    root = tk.Tk()
    root.title("Muon G-2 Kernel Density Aproximator")
    root.geometry("1920x1080")

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