import unittest

from test_classifier import TestBatchClassifier
from test_classifier import TestBatchClassifierBinary
from test_density import TestBatchDensityEstimator
from test_quantizer import TestBatchQuantizer
from test_transformer import TestBatchTransformer

from boostedmlp import TestBoostedMLP
from boostedmlp import TestStackedMLP
from boostedstumps import MiscStumpTests
from boostedstumps import TestBoostedStumps
from boostedstumps import TestStump1
from fastica import TestFastICA
from fmixtures import TestFastGaussianMixtureFixed
from gng import TestGrowingNeuralGasQuantizer
from kmeans import TestKMeansQuantizer
from kmeans import TestSlowKMeansQuantizer
from mixtures import TestGaussianMixture
from mixtures import TestGaussianMixtureFixed
from mlp import TestMLP
from pca import TestPCA
from pca import TestPCAGHA
from som import TestSOMQuantizer
from som import TestSOMTransformer

from heap import TestHeap
from heap import TestHeapSet

unittest.main()

