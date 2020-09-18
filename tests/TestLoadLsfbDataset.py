import unittest
from utils.lsfb_dataset_loader import load_lsfb_dataset

class TestLoadLsfbDataset(unittest.TestCase):

  def test_import_dataset(self):
    data = load_lsfb_dataset("./mock-data")
    self.assertEqual(len(data),35)

  def test_correct_subset_size(self):
    data = load_lsfb_dataset("./mock-data")
    test = data[data["subset"] == 'test']
    train = data[data["subset"] == 'train']

    self.assertEqual(len(test), 10)
    self.assertEqual(len(train), 25)

