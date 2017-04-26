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
        self.corpus_path = corpus_path
        self.corpus = pd.read_pickle(corpus_path+'/corpus_processed.pkl')

    def get_wavenet_batch(self, batch_size, length=64000):
        rows = self.corpus.shape[0]
        random_choice = np.random.randint(0,rows)
        row = self.corpus.iloc[random_choice]
        gc_input_vector = row['global_condition']
        gc_input_vector = gc_input_vector.reshape(1, 256)
        samples = utils.load_wav('/'.join([self.corpus_path, row['subPath'], self.corpus.index[random_choice]]), 16000)
        batchsizelen = (length*batch_size)
        chunk_offset = np.random.randint(0, samples.shape[0]-batchsizelen)
        samples = samples[chunk_offset:chunk_offset+batchsizelen]
        assert samples.shape[0] == batchsizelen
        samples = samples.reshape(batch_size, length)
        assert samples.shape[0] == batch_size
        assert samples.shape[1] == length
        return {"gc_in_vector": gc_input_vector, "wav": samples}