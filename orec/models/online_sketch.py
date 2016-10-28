from abc import ABCMeta, abstractmethod
from orec.recommender import Recommender

import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import safe_sparse_dot


class BaseProjection:

    """Base class for projection of context-aware matrix.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, k, p):
        """Initialize projection matrices.

        Args:
            k (int): Number of reduced dimensions (i.e. rows of projection mat.).
            p (int): Number of input dimensions (i.e. cols of projection mat.).
                p will be increased in the future due to new user/item/context insersion.

        """
        pass

    @abstractmethod
    def insert_proj_col(self, offset):
        """Insert a new column for a projection matrix.

        Args:
            offset (int): Index of the inserted column.

        """
        pass

    @abstractmethod
    def reduce(self, Y):
        """Make projection for an input matrix.

        Args:
            Y (numpy array; (p, n)): Input p-by-n matrix projected to a k-by-n matrix.

        Returns:
            numpy array; (k, n): Projected matrix.

        """
        return


class Raw(BaseProjection):

    def __init__(self, k, p):
        # k == p
        self.I = np.identity(p)

    def insert_proj_col(self, offset):
        pass

    def reduce(self, Y):
        return safe_sparse_dot(self.I, Y)


class RandomProjection(BaseProjection):

    def __init__(self, k, p):
        self.k = k
        self.R = sp.csr_matrix(self.__create_proj_mat((k, p)))

    def insert_proj_col(self, offset):
        col = self.__create_proj_mat((self.k, 1))
        R = self.R.toarray()
        self.R = sp.csr_matrix(np.concatenate((R[:, :offset], col, R[:, offset:]), axis=1))

    def reduce(self, Y):
        return safe_sparse_dot(self.R, Y)

    def __create_proj_mat(self, size):
        """Create a random projection matrix

        [1] D. Achlioptas. Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
        [2] P. Li, et al. Very sparse random projections.

        http://scikit-learn.org/stable/modules/random_projection.html#sparse-random-projection
        """

        # [1]
        # return np.random.choice([-np.sqrt(3), 0, np.sqrt(3)], size=size, p=[1 / 6, 2 / 3, 1 / 6])

        # [2]
        s = 1 / 0.2
        return np.random.choice([-np.sqrt(s / self.k), 0, np.sqrt(s / self.k)],
                                size=size,
                                p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])


class RandomMaclaurinProjection(BaseProjection):

    def __init__(self, k, p):
        self.k = k

        self.W1 = np.random.choice([1, -1], size=(k, p))
        self.W2 = np.random.choice([1, -1], size=(k, p))

    def insert_proj_col(self, offset):
        col = np.random.choice([1, -1], size=(self.k, 1))
        self.W1 = np.concatenate((self.W1[:, :offset], col, self.W1[:, offset:]), axis=1)

        col = np.random.choice([1, -1], size=(self.k, 1))
        self.W2 = np.concatenate((self.W2[:, :offset], col, self.W2[:, offset:]), axis=1)

    def reduce(self, Y):
        return safe_sparse_dot(self.W1, Y) * safe_sparse_dot(self.W2, Y) / np.sqrt(self.k)


class TensorSketchProjection(BaseProjection):

    def __init__(self, k, p):
        self.k = k

        self.h1 = np.random.choice(range(k), size=p)
        self.h2 = np.random.choice(range(k), size=p)
        self.h1_indices = [np.where(self.h1 == j)[0] for j in range(k)]
        self.h2_indices = [np.where(self.h2 == j)[0] for j in range(k)]
        self.s1 = np.random.choice([1, -1], size=p)
        self.s2 = np.random.choice([1, -1], size=p)

    def insert_proj_col(self, offset):
        self.h1 = np.concatenate((self.h1[:offset],
                                  np.random.choice(range(self.k), (1, )),
                                  self.h1[offset:]))
        self.h2 = np.concatenate((self.h2[:offset],
                                  np.random.choice(range(self.k), (1, )),
                                  self.h2[offset:]))
        self.h1_indices = [np.where(self.h1 == j)[0] for j in range(self.k)]
        self.h2_indices = [np.where(self.h2 == j)[0] for j in range(self.k)]
        self.s1 = np.concatenate((self.s1[:offset], np.random.choice([1, -1], (1, )), self.s1[offset:]))
        self.s2 = np.concatenate((self.s2[:offset], np.random.choice([1, -1], (1, )), self.s2[offset:]))

    def reduce(self, Y):
        if sp.isspmatrix(Y):
            Y = Y.toarray()

        sketch1, sketch2 = self.__sketch(Y)

        return np.real(np.fft.ifft(np.fft.fft(sketch1, axis=0) * np.fft.fft(sketch2, axis=0), axis=0))

    def __sketch(self, X):
        sketch1 = np.array([np.sum(np.array([self.s1[idx]]).T * X[idx], axis=0) for idx in self.h1_indices])
        sketch2 = np.array([np.sum(np.array([self.s2[idx]]).T * X[idx], axis=0) for idx in self.h2_indices])

        return sketch1, sketch2


class OnlineSketch(Recommender):

    """Inspired by: Streaming Anomaly Detection using Online Matrix Sketching
    """

    def __init__(self, contexts, k=40, ell=-1, r=-1, proj='Raw'):

        self.contexts = contexts
        self.p = np.sum(list(contexts.values()))

        # dimension of projected vectors
        # for `Raw` (i.e. w/o projection), k must equat to p
        self.k = self.p if proj == 'Raw' else k

        # if there is no preference for ell,
        # this will be sqrt(k) similarly to what the original streaming anomaly detection paper did
        self.ell = int(np.sqrt(self.k)) if ell < 1 else ell

        # number of tracked orthogonal bases
        # * upper bound of r is ell (r <= ell) because U_r is obtained from SVD(B) (or SVD(E)), and
        #   B and E always have ell columns
        self.r = int(np.ceil(self.ell / 2)) if r < 1 else np.min(r, self.ell)

        # initialize projection instance which is specified by `proj` argument
        constructor = globals()[proj]
        self.proj = constructor(self.k, self.p)

        self.clear()

    def clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.i_mat = sp.csr_matrix([])

    def check(self, d):

        u_index = d['u_index']
        is_new_user = u_index not in self.users
        if is_new_user:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

        i_index = d['i_index']
        is_new_item = i_index not in self.items
        if is_new_item:
            self.items[i_index] = {}
            self.n_item += 1

            i_vec = sp.csr_matrix(np.array([d['item']]).T)
            if self.i_mat.size == 0:
                self.i_mat = i_vec
            else:
                self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, i_vec)))

        return is_new_user, is_new_item

    def update(self, d, is_batch_train=False):
        y = np.concatenate((d['user'], d['others'], d['item']))
        y = self.proj.reduce(np.array([y]).T)
        y = np.ravel(preprocessing.normalize(y, norm='l2', axis=0))

        if not hasattr(self, 'B'):
            self.B = np.zeros((self.k, self.ell))

        # combine current sketched matrix with input at time t
        zero_cols = np.nonzero([np.isclose(s_col, 0.0) for s_col in np.sum(self.B, axis=0)])[0]
        j = zero_cols[0] if zero_cols.size != 0 else self.ell - 1  # left-most all-zero column in B
        self.B[:, j] = y

        U, s, V = ln.svd(self.B, full_matrices=False)

        # update the tracked orthonormal bases
        self.U_r = U[:, :self.r]

        # update ell orthogonal bases
        U_ell = U[:, :self.ell]
        s_ell = s[:self.ell]

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s_ell[-1] ** 2
        s_ell = np.sqrt(s_ell ** 2 - delta)

        self.B = np.dot(U_ell, np.diag(s_ell))

    def recommend(self, d, target_i_indices):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, target_i_indices]

        n_target = len(target_i_indices)

        # u_mat will be (n_user_context, n_item) for the target user
        u = np.concatenate((d['user'], d['others']))
        u_vec = np.array([u]).T

        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        Y = sp.vstack((u_mat, i_mat))
        Y = self.proj.reduce(Y)
        Y = sp.csr_matrix(preprocessing.normalize(Y, norm='l2', axis=0))

        X = np.identity(self.k) - np.dot(self.U_r, self.U_r.T)
        A = safe_sparse_dot(X, Y, dense_output=True)

        scores = ln.norm(A, axis=0, ord=2)

        return self.scores2recos(scores, target_i_indices)


class OnlineRandomSketch(OnlineSketch):

    """Inspired by: Streaming Anomaly Detection using Randomized Matrix Sketching
    [WIP] many matrix multiplications are computational heavy
    """

    def update(self, d, is_batch_train=False):
        y = np.concatenate((d['user'], d['others'], d['item']))
        y = self.proj.reduce(np.array([y]).T)
        y = np.ravel(preprocessing.normalize(y, norm='l2', axis=0))

        if not hasattr(self, 'E'):
            self.E = np.zeros((self.k, self.ell))

        # combine current sketched matrix with input at time t
        zero_cols = np.nonzero([np.isclose(s_col, 0.0) for s_col in np.sum(self.E, axis=0)])[0]
        j = zero_cols[0] if zero_cols.size != 0 else self.ell - 1  # left-most all-zero column in B
        self.E[:, j] = y

        O = np.random.normal(0., 0.1, (self.k, 100 * self.ell))
        MM = np.dot(self.E, self.E.T)
        Q, R = ln.qr(np.dot(MM, O))

        # eig() returns eigen values/vectors with unsorted order
        s, A = ln.eig(np.dot(np.dot(Q.T, MM), Q))
        order = np.argsort(s)[::-1]
        s = s[order]
        A = A[:, order]

        U = np.dot(Q, A)

        # update the tracked orthonormal bases
        self.U_r = U[:, :self.r]

        # update ell orthogonal bases
        U_ell = U[:, :self.ell]
        s_ell = s[:self.ell]

        # shrink step in the Frequent Directions algorithm
        delta = s_ell[-1]
        s_ell = np.sqrt(s_ell - delta)

        self.E = np.dot(U_ell, np.diag(s_ell))


class OnlineSparseSketch(OnlineSketch):

    """Inspired by: Efficient Frequent Directions Algorithm for Sparse Matrices
    """

    def update(self, d, is_batch_train=False):
        y = np.concatenate((d['user'], d['others'], d['item']))
        y = self.proj.reduce(np.array([y]).T)
        y = preprocessing.normalize(y, norm='l2', axis=0)  # (k, 1)

        if not hasattr(self, 'B'):
            self.p_failure = 0.1
            self.B = np.zeros((self.k, self.ell))
            self.A = np.array([])

        U, s, V = ln.svd(self.B, full_matrices=False)

        # update the tracked orthonormal bases
        self.U_r = U[:, :self.r]

        if self.A.size == 0:
            self.A = np.empty_like(y)
            self.A[:] = y[:]
        else:
            self.A = np.concatenate((self.A, y), axis=1)

        if np.count_nonzero(self.A) >= (self.ell * self.k) or self.A.shape[1] == self.k:
            B = self.__boosted_sparse_shrink(self.A, self.ell, self.p_failure)
            self.B = self.__dense_shrink(np.concatenate((self.B, B), axis=1), self.ell)
            self.A = np.array([])

    def __simultaneous_iteration(self, A, k, eps):
        n, d = A.shape

        q = int(np.log((n / eps) / eps))
        G = np.random.normal(0., 1., (d, k))

        # Gram-Schmidt
        Y = np.dot(np.dot(ln.matrix_power(np.dot(A, A.T), q), A), G)  # (n, k)
        Q, R = ln.qr(Y, mode='complete')

        return Q

    def __sparse_shrink(self, A, ell):
        m, d = A.shape

        Z = self.__simultaneous_iteration(A, ell, 1 / 4)
        P = sp.csr_matrix(np.dot(Z.T, A))

        # 1 <= ell < min(P.shape)
        H_ell, s_ell, V_ell = sp.linalg.svds(P, ell)

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        l = s_ell[-1] ** 2
        s_ell = np.sqrt(s_ell ** 2 - l)

        return np.dot(H_ell, np.diag(s_ell))

    def __boosted_sparse_shrink(self, A, ell, p_failure):
        al = (6 / 41) * ell
        sq_norm_A = ln.norm(A, ord='fro') ** 2
        AA = np.dot(A, A.T)

        while True:
            B = self.__sparse_shrink(A, ell)
            delta = (sq_norm_A - (ln.norm(B, ord='fro') ** 2)) / al

            C = (AA - np.dot(B, B.T)) / (delta / 2)
            if self.__verify_spectral(C, p_failure):
                return B

    def __dense_shrink(self, A, ell):
        H, s, V = ln.svd(A, full_matrices=False)

        H_ell = H[:, :ell]
        s_ell = s[:ell]

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        l = s_ell[-1] ** 2
        s_ell = np.sqrt(s_ell ** 2 - l)

        return np.dot(H_ell, np.diag(s_ell))

    def __verify_spectral(self, C, p_failure):
        d = C.shape[0]
        c = 1  # some constant

        if not hasattr(self, 'i_verify'):
            self.i_verify = 0

        self.i_verify += 1
        p_failure_i = p_failure / (2 * (self.i_verify ** 2))

        # pick a point uniformly at ranom from the d-dimensional unit sphere
        # http://stackoverflow.com/questions/15880367/python-uniform-distribution-of-points-on-4-dimensional-sphere
        normal_deviates = np.random.normal(size=(d, 1))
        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        x = normal_deviates / radius

        return ln.norm(np.dot(ln.matrix_power(C, int(c * np.log(d / p_failure_i))), x)) <= 1
