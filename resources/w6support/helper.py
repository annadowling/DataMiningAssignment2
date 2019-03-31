"""
Utility functions that are shared across notebooks but not shown
"""

import pylab as pl
import numpy as np
import itertools

def plot_2d_class(X, y, nTrain, model, plotTitle, fileTitle, cmap_area, cmap_pts):
    predNames=list(X.columns)
    c1 = predNames[:1] # first of 2
    c2 = predNames[-1:] # last of 2
    x_min, x_max = X[c1].min() - .1, X[c1].max() + .1
    y_min, y_max = X[c2].min() - .1, X[c2].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Add the areas coloured by the fit
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_area)

    x1 = list(itertools.chain.from_iterable(X[c1].values)) # https://stackoverflow.com/a/11264799
    x2 = list(itertools.chain.from_iterable(X[c2].values))

    # Plot the points with colours in the same colour segment as the area
    pl.scatter(x1[:nTrain], x2[:nTrain], c=y[:nTrain], cmap=cmap_pts) # training data
    pl.scatter(x1[nTrain:], x2[nTrain:], c=y[nTrain:], cmap=cmap_pts, edgecolors="black") # test data

    pl.title(plotTitle)
    pl.xlabel(c1[0])
    pl.ylabel(c2[0])
    pl.axis('tight')
    pl.savefig(fileTitle)

