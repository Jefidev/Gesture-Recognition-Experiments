import unittest
from models.VideoRNN import VideoRNN

class TestBaseLoader(unittest.TestCase):

  def test_model_ini(self):
    model = VideoRNN(4096, 2048, 10)