'''
 In order to sample from a random distribution, we plot the function and a random point (x,y) and if the point is under the curve, return the point
'''
import random 
import math
import numpy as np


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

