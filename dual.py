import numpy as np

from numpy import linalg
import cvxopt
import cvxopt.solvers

# Plotting lib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Saving images
import os
from datetime import datetime

### KERNELS ###

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


### DUAL SVM ###

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))



def dual(X, y, kernel=linear_kernel, C=1, figsize=[15,15],
	dotsize=150, padding=0.1, color_bar=False, verbose=False, save_path=None, fontsize=15):

	# data
	X_train = np.array(X).astype('float')
	y_train = np.array(y).astype('float')

    # Fit and predict

	clf = SVM(C=C, kernel=kernel)
	clf.fit(X_train, y_train)
    
    # accuracy report on training set
	y_predict = clf.predict(X_train)
	correct = np.sum(y_predict == y_train)
	print("training accuracy: %d out of %d" % (correct, len(y_predict)))

	# Plot

    # plot sample density
	p = 1

	mask = np.random.choice([True, False], size=X_train.shape[0],
		p=[p, 1.0 - p])

	features = X_train[mask]
	targets = y_train[mask]

	features_x = X_train[:, 0]
	features_y = X_train[:, 1]

	x_lim = [features_x.min() - padding, features_x.max() + padding]
	y_lim = [features_y.min() - padding, features_y.max() + padding]

	plt.figure(figsize=figsize)


	XX, YY = np.mgrid[x_lim[0]:x_lim[1]:100j, y_lim[0]:y_lim[1]:100j]
	ravel = np.c_[XX.ravel(), YY.ravel()]
	Z = clf.project(ravel).reshape(XX.shape)

	cmbg = LinearSegmentedColormap.from_list('color_map_bg',
		[(16/255, 90/255, 173/255), (193/255, 110/255, 49/255)], N=100)
	cmdot = LinearSegmentedColormap.from_list('color_map_dot',
		[(11/255, 165/255, 254/255), (255/255, 150/255, 0/255)], N=100)
	plt.pcolormesh(XX, YY, Z, antialiased=True, cmap=cmbg)

	if (color_bar):
		plt.colorbar()

	X1_train = X_train[y_train==1]
	X2_train = X_train[y_train==-1]

    # Draw the rest of the features
	values = (targets[:] == 1).astype(np.float)
	plt.scatter(features_x, features_y, c=values, zorder=10,
				edgecolors='white',s=dotsize, linewidths=1.5, cmap=cmdot)

    # Highlight Support vectors
	plt.scatter(clf.sv[:,0], clf.sv[:,1], s=dotsize,
					facecolors='none', edgecolors='white', linewidths=3,cmap=cmdot)

    # Margin Contour
	plt.contour(XX, YY, Z, [0.0], colors='white', linewidths=1, origin='lower')
	plt.contour(XX, YY, Z + 1, [0.0], colors='white', linewidths=1, origin='lower', linestyles='dashed')
	plt.contour(XX, YY, Z - 1, [0.0], colors='white', linewidths=1, origin='lower', linestyles='dashed')

	plt.xlim(x_lim)
	plt.ylim(y_lim)
	
	try:
		plt.xlabel(str(X.columns[0]), fontsize=fontsize)
		plt.ylabel(str(X.columns[1]), fontsize=fontsize)
	except:
		pass
	title = str(kernel.__name__) + "\nC = " + str(C)
	plt.title(title, fontsize=fontsize)
	plt.grid()

	if not os.path.exists("images"):
		os.mkdir("images")

	if (save_path != None):
		now = datetime.now()
		date_time = now.strftime("%m%d%Y%H%M%S")
		plt.savefig(save_path + "dual_" + date_time + "_" + str(C) + "_" + str(kernel.__name__) + ".png")

	plt.show()




