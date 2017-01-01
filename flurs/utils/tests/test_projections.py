from unittest import TestCase
import numpy as np

from flurs.utils.projections import Raw, RandomProjection, RandomMaclaurinProjection, TensorSketchProjection


class ProjectionsTestCase(TestCase):

    def setUp(self):
        self.Y = np.arange(100).reshape((50, 2))
        self.p = self.Y.shape[0]
        self.k = int(np.sqrt(self.p))

    def test_raw(self):
        proj = Raw(self.k, self.p)
        Y_ = proj.reduce(self.Y)
        self.assertEqual(Y_.shape, self.Y.shape)

    def test_random(self):
        proj = RandomProjection(self.k, self.p)
        Y_ = proj.reduce(self.Y)
        self.assertEqual(Y_.shape, (self.k, self.Y.shape[1]))

    def test_random_maclaurin(self):
        proj = RandomMaclaurinProjection(self.k, self.p)
        Y_ = proj.reduce(self.Y)
        self.assertEqual(Y_.shape, (self.k, self.Y.shape[1]))

    def test_tensor_sketch(self):
        proj = TensorSketchProjection(self.k, self.p)
        Y_ = proj.reduce(self.Y)
        self.assertEqual(Y_.shape, (self.k, self.Y.shape[1]))
