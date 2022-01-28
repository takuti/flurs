"""Utility classes for vector projection.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class BaseProjection(object):

    """Base class for projection of feature vectors.

    Parameters
    ----------
    k : int
        Number of reduced dimensions (i.e. rows of projection mat.).

    p : int
        Number of input dimensions (i.e. cols of projection mat.).
        ``p`` will be increased in the future due to new user/item/context insersion.
    """

    def __init__(self, k, p):
        pass

    def insert_proj_col(self, offset):
        """Insert a new column into a projection matrix.

        Parameters
        ----------
        offset : int
            Column index of where the new dimension is inserted
        """
        pass

    def reduce(self, Y):
        """Apply projection to an input matrix.

        Parameters
        ----------
        Y : numpy array, (p, n)
            Input p-by-n matrix projected to a k-by-n matrix.

        Returns
        -------
        array
            A k-by-n projected matrix.
        """
        return


class Raw(BaseProjection):

    """Raw projector that does nothing regardless of the parameters
    i.e., a projector is an identity matrix.

    Parameters
    ----------
    k : int
        Number of reduced dimensions (i.e. rows of projection mat.).

    p : int
        Number of input dimensions (i.e. cols of projection mat.).
        ``p`` will be increased in the future due to new user/item/context insersion.
    """

    def __init__(self, k, p):
        # k == p
        self.E = np.identity(p)

    def insert_proj_col(self, offset):
        pass

    def reduce(self, Y):
        return safe_sparse_dot(self.E, Y)


class RandomProjection(BaseProjection):

    """Projection based on a randomly initialized matrix.

    Parameters
    ----------
    k : int
        Number of reduced dimensions (i.e. rows of projection mat.).

    p : int
        Number of input dimensions (i.e. cols of projection mat.).
        ``p`` will be increased in the future due to new user/item/context insersion.

    density : float, default=0.2
        Density parameter used to create a projection matrix.

    References
    ----------
    .. [1] D. Achlioptas. Database-friendly random projections:
           Johnson-Lindenstrauss with binary coins.
    .. [2] P. Li, et al. Very sparse random projections.
    .. [3] `Scikit-learn documentation: Sparse random projection
           <http://scikit-learn.org/stable/modules/random_projection.html#sparse-random-projection>`_.
    """

    def __init__(self, k, p, density=0.2):
        self.k = k
        self.density = density
        self.R = sp.csr_matrix(self.__create_proj_mat((k, p)))

    def insert_proj_col(self, offset):
        col = self.__create_proj_mat((self.k, 1))
        R = self.R.toarray()
        self.R = sp.csr_matrix(
            np.concatenate((R[:, :offset], col, R[:, offset:]), axis=1)
        )

    def reduce(self, Y):
        return safe_sparse_dot(self.R, Y)

    def __create_proj_mat(self, size):
        # [1]_
        # return np.random.choice([-np.sqrt(3), 0, np.sqrt(3)],
        #                         size=size, p=[1 / 6, 2 / 3, 1 / 6])

        # [2]_
        s = 1 / self.density
        return np.random.choice(
            [-np.sqrt(s / self.k), 0, np.sqrt(s / self.k)],
            size=size,
            p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)],
        )


class RandomMaclaurinProjection(BaseProjection):

    """Random Maclaurin Projection.

    Parameters
    ----------
    k : int
        Number of reduced dimensions (i.e. rows of projection mat.).

    p : int
        Number of input dimensions (i.e. cols of projection mat.).
        ``p`` will be increased in the future due to new user/item/context insersion.
    """

    def __init__(self, k, p):
        self.k = k

        self.W1 = np.random.choice([1, -1], size=(k, p))
        self.W2 = np.random.choice([1, -1], size=(k, p))

    def insert_proj_col(self, offset):
        col = np.random.choice([1, -1], size=(self.k, 1))
        self.W1 = np.concatenate(
            (self.W1[:, :offset], col, self.W1[:, offset:]), axis=1
        )

        col = np.random.choice([1, -1], size=(self.k, 1))
        self.W2 = np.concatenate(
            (self.W2[:, :offset], col, self.W2[:, offset:]), axis=1
        )

    def reduce(self, Y):
        return (
            safe_sparse_dot(self.W1, Y) * safe_sparse_dot(self.W2, Y) / np.sqrt(self.k)
        )


class TensorSketchProjection(BaseProjection):

    """Tensor Sketch Projection.

    Parameters
    ----------
    k : int
        Number of reduced dimensions (i.e. rows of projection mat.).

    p : int
        Number of input dimensions (i.e. cols of projection mat.).
        ``p`` will be increased in the future due to new user/item/context insersion.
    """

    def __init__(self, k, p):
        self.k = k

        self.h1 = np.random.choice(range(k), size=p)
        self.h2 = np.random.choice(range(k), size=p)
        self.h1_indices = [np.where(self.h1 == j)[0] for j in range(k)]
        self.h2_indices = [np.where(self.h2 == j)[0] for j in range(k)]
        self.s1 = np.random.choice([1, -1], size=p)
        self.s2 = np.random.choice([1, -1], size=p)

    def insert_proj_col(self, offset):
        self.h1 = np.concatenate(
            (self.h1[:offset], np.random.choice(range(self.k), (1,)), self.h1[offset:])
        )
        self.h2 = np.concatenate(
            (self.h2[:offset], np.random.choice(range(self.k), (1,)), self.h2[offset:])
        )
        self.h1_indices = [np.where(self.h1 == j)[0] for j in range(self.k)]
        self.h2_indices = [np.where(self.h2 == j)[0] for j in range(self.k)]
        self.s1 = np.concatenate(
            (self.s1[:offset], np.random.choice([1, -1], (1,)), self.s1[offset:])
        )
        self.s2 = np.concatenate(
            (self.s2[:offset], np.random.choice([1, -1], (1,)), self.s2[offset:])
        )

    def reduce(self, Y):
        if sp.isspmatrix(Y):
            Y = Y.toarray()

        sketch1, sketch2 = self.__sketch(Y)

        return np.real(
            np.fft.ifft(
                np.fft.fft(sketch1, axis=0) * np.fft.fft(sketch2, axis=0), axis=0
            )
        )

    def __sketch(self, X):
        sketch1 = np.array(
            [
                np.sum(np.array([self.s1[idx]]).T * X[idx], axis=0)
                for idx in self.h1_indices
            ]
        )
        sketch2 = np.array(
            [
                np.sum(np.array([self.s2[idx]]).T * X[idx], axis=0)
                for idx in self.h2_indices
            ]
        )

        return sketch1, sketch2
