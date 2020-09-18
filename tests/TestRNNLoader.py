import unittest
from utils.lsfb_dataset_loader import load_lsfb_dataset
from batch_loaders.RNNBatchLoader import RNNBatchLoader

class TestRNNLoader(unittest.TestCase):

  def setUp(self):
    self.data = load_lsfb_dataset("./mock-data", verbose=False)

  def test_get_batch(self):
    batch_size = 10
    loader = RNNBatchLoader(self.data, batch_size, (244,244))

    for batch in range(len(loader)):
      X, y = loader.get_batch(0)
      self.assertEqual(batch_size, len(y))

