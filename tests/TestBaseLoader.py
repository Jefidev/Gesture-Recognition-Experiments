import unittest
from utils.lsfb_dataset_loader import load_lsfb_dataset
from batch_loaders.BaseLoader import BaseLoader

class TestBaseLoader(unittest.TestCase):

  def setUp(self):
    self.data = load_lsfb_dataset("./mock-data", verbose=False)

  def test_get_label_mapping(self):
    loader = BaseLoader(self.data, 10, (240,240))

    mapping = loader.get_label_mapping()
    self.assertEqual(len(mapping), 5)

  def test_get_batch(self):
    loader = BaseLoader(self.data, 10, (240,240))

    with self.assertRaises(NotImplementedError) as context:
      loader.get_batch(2)

  def test_get_len(self):
    loader = BaseLoader(self.data, 10, (240,240))

    self.assertEqual(3, len(loader))
