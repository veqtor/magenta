"""Module to load the Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf
import pandas as pd

from magenta.models.nsynth import utils

class MusicPostBotDataset(object):
    def __init__(self, corpus_path, is_training=True):
        self.is_training = is_training
        self.corpus = corpus_path

    def get_wavenet_batch(self, batch_size, length=64000):

        gc_input_vector = np.linspace(0,255, num=256)
        gc_input_vector = gc_input_vector.reshape(-1, 256)
        gc_input_vector = gc_input_vector.repeat(repeats=batch_size)
        gc_input_vector = gc_input_vector.reshape(-1, 256)

        samples = np.linspace(-1, 1, num=length)
        samples = samples.reshape(-1, length)
        samples = samples.repeat(repeats=batch_size)
        samples = samples.reshape(-1, length)


        return {"gc_in_vector": gc_input_vector, "wav": samples}
