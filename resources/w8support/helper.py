"""
Utility functions that are shared across notebooks but not shown
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import matplotlib.mlab as mlab
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
import scipy.stats as stats


def fitClusterLabels(data, algorithm, args, kwds):
    model = algorithm(*args, **kwds)
    if (str(algorithm.__name__) == 'GaussianMixture'):
        labels = model.fit(data).predict(data)
    else:
        labels = model.fit_predict(data)
    return model, labels	

def plot_2dClusters(data, labels, title, paletteName, fontSize, plot_kwds):
	# Based on 
    sns.set_context('poster')
    sns.set_color_codes()
    palette = sns.color_palette(paletteName, np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(title, fontsize=fontSize)
    return plt

def plot_clusters(data, algorithm, args, kwds, plot_kwds):
	# Based on 
    sns.set_context('poster')
    sns.set_color_codes()
    start_time = time.time()
    if (str(algorithm.__name__) == 'GaussianMixture'):
        labels = algorithm(*args, **kwds).fit(data).predict(data)
    else:
        labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    elapsed = end_time-start_time
    return plt, elapsed
        
def overlayEllipses(plt, weights, means, covariances, **kwargs):
    ax = plt.gca()
    w_factor = 0.2 / weights.max()
    k = len(weights)
    for position, covariance, w in zip(means, covariances, weights):
        alpha = w*w_factor
    
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        elif covariance.shape == (2,1):
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        else:
            angle = 0
            width = height = 2 * np.sqrt(covariance)
    
        # Draw the Ellipse
        for nsig in range(1, k):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

    return plt

def overlayDisks(plt, centres, radii, fc, plot_kwds):
    frame = plt.gca()
    alpha = plot_kwds.get('alpha')
    for c, r in zip(centres, radii):
#        frame.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.3, zorder=1))
        frame.add_patch(plt.Circle(c, r, fc=fc, lw=3, alpha=alpha, zorder=1))
    return plt

def plot_2dClustersWithEllipses(X, labels, weights, means, covariances, ax=None):
    """ See PythonDataScienceHandbook/notebooks/05.12-Gaussian-Mixtures.ipynb """ 
    ax = ax or plt.gca()
    if (labels != None):
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / weights.max()
    for pos, covar, w in zip(means, covariances, weights):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def generate_MoG_data(num_data, means, covariances, weights):
    """ Creates a list of data points
	Based on Expectation-Maximization-Gaussian-Mixtures/EM-for-gmm.ipynb """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data
    
def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z.
    Based on Expectation-Maximization-Gaussian-Mixtures/EM-for-gmm.ipynb """
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters.
    Based on Expectation-Maximization-Gaussian-Mixtures/EM-for-gmm.ipynb """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll

def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for i in range(maxiter):
        if i % 5 == 0:
            print("Iteration %s" % i)
        
        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        # Hint: To compute likelihood of seeing data point j given cluster k, use multivariate_normal.pdf.
        for j in range(num_data):
            for k in range(num_clusters):
                resp[j, k] =  weights[k] * stats.multivariate_normal.pdf(data[j], mean=means[k], cov=covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums # normalize over all possible cluster assignments

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = np.sum(resp, axis=0)
        
        for k in range(num_clusters):
            
            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            weights[k] = counts[k]
            
            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            
            weighted_sum = 0
            for j in range(num_data):
                weighted_sum += data[j] * resp[j,k]
            means[k] = weighted_sum / weights[k]
            
            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                #(Hint: Use np.outer on the data[j] and this cluster's mean)
                weighted_sum += np.outer(data[j] - means[k],data[j] - means[k]) * resp[j,k]

            covariances[k] = weighted_sum / weights[k]

        
        
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest
    
    if i % 5 != 0:
        print("Iteration %s" % i)
    weights = weights / sum(weights)
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out

def plot_contours(data, means, covs, title, paletteName, fontSize):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo', 'yellow', 'purple']
    #palette = sns.color_palette(paletteName, k + 1)
    #colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
#        plt.contour(X, Y, Z, colors = palette[i])
        plt.title(title)
    plt.rcParams.update({'font.size':fontSize})
    plt.tight_layout()
    
def find_kMeans_clusters(X, n_clusters, rseed=2):
    """ From  PythonDataScienceHandbook/notebooks/05.11-K-Means.ipynb
    1. Randomly choose clusters """
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels
    
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    """ See PythonDataScienceHandbook/notebooks/05.12-Gaussian-Mixtures.ipynb """ 
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """ See PythonDataScienceHandbook/notebooks/05.12-Gaussian-Mixtures.ipynb 
    Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    elif covariance.shape == (2,1):
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    """ See PythonDataScienceHandbook/notebooks/05.12-Gaussian-Mixtures.ipynb """ 
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def plot_digits(data):
    """ See PythonDataScienceHandbook/notebooks/05.12-Gaussian-Mixtures.ipynb """ 
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

