from kde2d import kde2D, kde2D_b, custom_kde2d
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum
from IPython.display import display, clear_output


#ENUM for all of the input types


COSYInputName = ["time (ns)", "r (mm)", "y (mm)", "MoM (Gev)"]

class COSYInputType(IntEnum):
     TIME = 0
     R = 1
     Y = 2
     MoM = 3


class CosyPlot:
    inputFile = "COSYinput_modified.txt"
    fig = []
    axes = []

    def __init__(self, f, a):
        self.fig = f
        self.axes = a


    def getCOSYSamples(self, x=-1, y=-1):   
            data1 = []
            data2 = []
            f = open(self.inputFile, 'r')
            lines = f.readlines()
            for i in range (len(lines)):
                if (i == 0):
                    continue
                data = lines[i].split(" ")
                if x == -1 or y == -1:
                    x = 0
                    y = 0
                data1.append(float(data[x]))
                data2.append(float(data[y]))
            return data1, data2

    # Create an empty plot with labels
    def plot_empty(self, inputTypeX : COSYInputType, inputTypeY : COSYInputType):
        self.axes[inputTypeX, inputTypeY].set_xlabel(COSYInputName[inputTypeX])
        self.axes[inputTypeX, inputTypeY].set_ylabel(COSYInputName[inputTypeY])
        plt.plot()


    def plot_all(self, bandwidth=1, resolution=100j):
        for i in range(4):
            for j in range(4):
                self.plot(i, j, bandwidth=bandwidth, resolution=resolution)

    def plot(self, inputTypeX : COSYInputType, inputTypeY : COSYInputType, xBand, yBand, resolution=100j, showPoints=True):
        self.axes[inputTypeX, inputTypeY].clear()

        xData, yData = self.getCOSYSamples(inputTypeX, inputTypeY)
        
        x_vals = np.linspace(min(xData), max(xData), 100)
        y_vals = np.linspace(min(yData), max(yData), 100)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        # Estimate the KDE with a given bandwidth
        density = custom_kde2d(x_grid, y_grid, xData, yData, xBand, yBand)
        self.axes[inputTypeX, inputTypeY].set_xlim([min(xData), max(xData)])
        self.axes[inputTypeX, inputTypeY].set_ylim([min(yData), max(yData)])
        self.axes[inputTypeX, inputTypeY].pcolormesh(x_grid, y_grid, density)
        if showPoints:
            self.axes[inputTypeX, inputTypeY].scatter(xData, yData, s=2, facecolor='gray')

        kdeMean, kdeStd = compute_kde_mean_and_std(x_grid, y_grid, density)
        dataMean, dataStd = compute_data_mean_and_std(xData, yData)
        #kdeMean = get_magnitude(kdeMean)
        #kdeStd = get_magnitude(kdeStd)
        #dataMean = get_magnitude(dataMean)
        #dataStd = get_magnitude(dataStd)
        self.axes[inputTypeX, inputTypeY].set_xlim([min(xData), max(xData)])
        self.axes[inputTypeX, inputTypeY].set_ylim([min(yData), max(yData)])
        self.axes[inputTypeX, inputTypeY].set_xlabel(COSYInputName[inputTypeX] + f"\nKDE: Average X: {kdeMean[0]:.2f} Average Y: {kdeMean[1]:.2f} \nσ X: {kdeStd[0]:.2f} σ Y: {kdeStd[1]:.2f}\nData: Average X: {dataMean[0]:.2f} Average Y: {dataMean[1]:.2f} \nσ X: {dataStd[0]:.2f} σ Y: {dataStd[1]:.2f}")
        self.axes[inputTypeX, inputTypeY].set_ylabel(COSYInputName[inputTypeY])

        plt.show()
        '''
        #inputTypeX = int(inputTypeX)
        #inputTypeY = int(inputTypeY)
        xData, yData = self.getCOSYSamples(inputTypeX, inputTypeY)
        xx, yy, zz = kde2D_b(xData, yData, xBand, yBand, resolution, resolution) 
        
        kdeMean, kdeStd = compute_kde_mean_and_std(xx, yy, zz)
        dataMean, dataStd = compute_data_mean_and_std(xData, yData)
        kdeMean = get_magnitude(kdeMean)
        kdeStd = get_magnitude(kdeStd)
        dataMean = get_magnitude(dataMean)
        dataStd = get_magnitude(dataStd)
        c = self.axes[inputTypeX, inputTypeY].pcolormesh(xx, yy, zz)
        if showPoints:
            self.axes[inputTypeX, inputTypeY].scatter(xData, yData, s=2, facecolor='gray')
        self.axes[inputTypeX, inputTypeY].set_xlim([min(xData), max(xData)])
        self.axes[inputTypeX, inputTypeY].set_ylim([min(yData), max(yData)])
        self.axes[inputTypeX, inputTypeY].set_xlabel(COSYInputName[inputTypeX] + f"\nKDE: Average: {kdeMean:.2f} σ: {kdeStd:.2f}\nData: Average: {dataMean:.2f} σ: {dataStd:.2f}")
        self.axes[inputTypeX, inputTypeY].set_ylabel(COSYInputName[inputTypeY])
        #self.axes[inputTypeX, inputTypeY].text(0, 0, "Average: 2.103 σ: 1.23")
        #c##lear_output(wait=True)
        plt.plot()
        #display(self.fig)'''


def compute_kde_mean_and_std(xx, yy, zz):
    # Compute the total density
    total_density = np.sum(zz)

    # Calculate the mean for x
    mean_x = np.sum(xx * zz) / total_density
    # Calculate the mean for y
    mean_y = np.sum(yy * zz) / total_density

    # Calculate the variance for x
    variance_x = np.sum(zz * (xx - mean_x)**2) / total_density
    # Calculate the variance for y
    variance_y = np.sum(zz * (yy - mean_y)**2) / total_density

    # Standard deviations
    std_x = np.sqrt(variance_x)
    std_y = np.sqrt(variance_y)

    return (mean_x, mean_y), (std_x, std_y)


def compute_data_mean_and_std(xData, yData):
    # Calculate mean
    mean_x = np.mean(xData)
    mean_y = np.mean(yData)

    # Calculate standard deviation
    std_x = np.std(xData)
    std_y = np.std(yData)

    return (mean_x, mean_y), (std_x, std_y)

def get_magnitude(vec):
    return np.sqrt(vec[0]**2+vec[1]**2)