# Import Modules
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math


def fit_plane():
    # some 3-dim points
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
    data = np.random.multivariate_normal(mean, cov, 50)

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
    XX = X.flatten()
    YY = Y.flatten()

    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
    temp = math.sqrt((C[0] * C[0] + C[1] * C[1] + 1))
    unit_vector = [-C[0] / temp, -C[1] / temp, 1 / temp]

    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]  # Or: Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Define Directory

file = 'C:\\Users\\JahandarH\\PycharmProjects\\KukaControl\\PlateDigitizations\\test_pointer_planes_t1.trc'
# Import Files

# Check the Input Data (Each file has 4 sets of xyz, with each reading including at least 3 markers)

# Import the data:
data = np.genfromtxt(file, delimiter=None, names=None, skip_header=6)

# Find the probe tip based on available markers at each reading
print("Based on the available markers, we should determine a probe tip")

# Fit a Plane to each data set
print("Fit a plane to all the probe points")
fit_plane()

# Check the plane Normal Vectors

# Check the planes intersection
print("Intersection of Planes 1,2 &3 should be 10 \" away from the intersection of 2,3 &5")
