# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A WaveNet-style AutoEncoder Configuration."""

# internal imports
import tensorflow as tf

from magenta.models.nsynth import reader
from magenta.models.nsynth import mpb_dset_reader
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import masked


class Config(object):
  """Configuration object that helps manage the graph."""

  def __init__(self, train_path=None):
    self.num_iters = 200000
    self.learning_rate_schedule = {
        0: 2e-4,
        90000: 4e-4 / 3,
        120000: 6e-5,
        150000: 4e-5,
        180000: 2e-5,
        210000: 6e-6,
        240000: 2e-6,
    }
    self.ae_hop_length = 512
    self.gc_input_width = 256
    self.ae_bottleneck_width = 32
    self.gc_bottleneck_width = 64
    self.receptive_field_size = 1536
    self.train_path = train_path

  def get_batch(self, batch_size):
    assert self.train_path is not None
    data_train = mpb_dset_reader.MusicPostBotDataset(self.train_path, is_training=True)
    return data_train.get_wavenet_batch(batch_size, length=6144)

  @staticmethod
  def _condition(x, encoding):
    """Condition the input on the encoding.

    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.

    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """
    mb, length, channels = x.get_shape().as_list()
    enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
    assert enc_mb == mb
    assert enc_channels == channels

    encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
    x = tf.reshape(x, [mb, enc_length, -1, channels])
    x += encoding
    x = tf.reshape(x, [mb, length, channels])
    x.set_shape([mb, length, channels])
    return x

  def build(self, inputs, is_training, is_generation=False):
    """Build the graph for this configuration.

    Args:
      inputs: A dict of inputs. For training, should contain 'wav'.
      is_training: Whether we are training or not. Not used in this config.

    Returns:
      A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
      the 'quantized_input', and whatever metrics we want to track for eval.
    """
    num_stages = 10
    num_layers = 30
    filter_length = 3
    res_width = 256
    skip_width = 256
    ae_num_stages = 10
    ae_num_layers = 30
    ae_filter_length = 3
    ae_width = 64

    # Encode the source with 8-bit Mu-Law.

    x = inputs['wav']
    x_quantized = utils.mu_law(x)
    x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
    x_scaled = tf.expand_dims(x_scaled, 2)
    x_mb, x_length, x_channels = x_scaled.get_shape().as_list()

    if is_generation:
        local_condition = inputs['en']
        x_scaled = tf.reshape(x_scaled,[1, -1, x_channels])
        x_mb, x_length, x_channels = x_scaled.get_shape().as_list()

        gc = inputs['gc']
        gc = tf.cast(gc, tf.float32)

        gc = tf.reshape(gc, [self.gc_bottleneck_width])
        gc = tf.tile(gc, [x_length])
        gc = tf.reshape(gc, [x_mb, x_length, self.gc_bottleneck_width])
    else:
        ###
        # The Global Encoder
        ###
        gc = inputs['gc_in_vector']
        gc = tf.cast(gc, tf.float32)
        gc = tf.reshape(gc, [-1, self.gc_input_width])
        gc = utils.dense_ch(gc, self.gc_input_width, self.gc_bottleneck_width * 4, scope='gc_ae_d1', is_training=is_training)
        gc = utils.dense_ch(gc, self.gc_bottleneck_width * 4, self.gc_bottleneck_width * 2, scope='gc_ae_d2', is_training=is_training)
        gc = utils.dense_ch(gc, self.gc_bottleneck_width * 2, self.gc_bottleneck_width, scope='gc_ae_d3', is_training=is_training, activation_fn=tf.nn.sigmoid)
        gc = tf.reshape(gc, [-1, 1, self.gc_bottleneck_width])

        ###
        # The Non-Causal Temporal Encoder.
        ###
        lc = masked.conv1d(
            x_scaled,
            causal=False,
            num_filters=ae_width,
            filter_length=ae_filter_length,
            name='ae_startconv')

        for num_layer in xrange(ae_num_layers):
          dilation = 2**(num_layer % ae_num_stages)
          d = tf.nn.relu(lc)
          d = masked.conv1d(
              d,
              causal=False,
              num_filters=ae_width,
              filter_length=ae_filter_length,
              dilation=dilation,
              name='ae_dilatedconv_%d' % (num_layer + 1))
          d = self._condition(d,
                              masked.conv1d(
                                  gc,
                                  num_filters=ae_width,
                                  filter_length=1,
                                  name='ae_gc_cond_map_%d' % (num_layer + 1)))
          d = tf.nn.relu(d)
          lc += masked.conv1d(
              d,
              num_filters=ae_width,
              filter_length=1,
              name='ae_res_%d' % (num_layer + 1))

        lc = masked.conv1d(
            lc,
            num_filters=self.ae_bottleneck_width,
            filter_length=1,
            name='ae_bottleneck')
        lc = masked.pool1d(lc, self.ae_hop_length, name='ae_pool', mode='avg')
        local_condition = lc

    ###
    # Local conditioning upsampler
    ###
    if is_generation:
        local_condition = tf.reshape(local_condition, [-1, x_length/self.ae_hop_length, self.ae_bottleneck_width])

    lc_mb, lc_length, lc_channels = local_condition.get_shape().as_list()

    assert self.ae_bottleneck_width == lc_channels
    assert x_mb == lc_mb
    print('lc_len: ' + str(int(lc_mb * self.ae_hop_length)) + ' xlen: ' + str(x_length) + ' x_mb: ' + str(x_mb) + ' xchans' + str(x_channels))

    local_condition = tf.reshape(local_condition, [-1, lc_mb, lc_length, lc_channels])

    local_condition = utils.conv2d(
        local_condition, [1, self.ae_hop_length / 4], [1, self.ae_hop_length / 4],
        self.ae_bottleneck_width,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="upsampler_1")
    local_condition = utils.conv2d(
        local_condition, [1, 4], [1, 4],
        self.ae_bottleneck_width,
        is_training,
        activation_fn=utils.leaky_relu(),
        transpose=True,
        batch_norm=True,
        scope="upsampler_2")

    local_condition = tf.reshape(local_condition, [x_mb, x_length, self.ae_bottleneck_width])

    uplc_mb, uplc_length, uplc_channels = local_condition.get_shape().as_list()

    assert uplc_mb == x_mb

    ###
    # The WaveNet Decoder.
    ###
    l = masked.shift_right(x_scaled)
    l = masked.conv1d(
        l, num_filters=res_width, filter_length=filter_length, name='startconv')

    # Set up skip connections.
    s = masked.conv1d(
        l, num_filters=skip_width, filter_length=1, name='skip_start')

    # Residual blocks with skip connections.
    for i in xrange(num_layers):
      dilation = 2**(i % num_stages)
      d = masked.conv1d(
          l,
          num_filters=2 * res_width,
          filter_length=filter_length,
          dilation=dilation,
          name='dilatedconv_%d' % (i + 1))
      d = self._condition(d,
                          masked.conv1d(
                              local_condition,
                              num_filters=2 * res_width,
                              filter_length=1,
                              name='cond_map_%d' % (i + 1)))
      d = self._condition(d,
                          masked.conv1d(
                              gc,
                              num_filters=2 * res_width,
                              filter_length=1,
                              name='gc_cond_map_%d' % (i + 1)))

      assert d.get_shape().as_list()[2] % 2 == 0
      m = d.get_shape().as_list()[2] // 2

      d_sigmoid = tf.sigmoid(d[:, :, :m])
      d_tanh = tf.tanh(d[:, :, m:])
      d = d_sigmoid * d_tanh

      l += masked.conv1d(
          d, num_filters=res_width, filter_length=1, name='res_%d' % (i + 1))
      s += masked.conv1d(
          d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
    s = self._condition(s,
                        masked.conv1d(
                            local_condition,
                            num_filters=skip_width,
                            filter_length=1,
                            name='cond_map_out1'))
    s = self._condition(s,
                        masked.conv1d(
                            gc,
                            num_filters=skip_width,
                            filter_length=1,
                            name='gc_cond_map_out1'))
    s = tf.nn.relu(s)

    ###
    # Compute the logits and get the loss.
    ###
    logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
    logits = tf.reshape(logits, [-1, 256])
    probs = tf.nn.softmax(logits, name='softmax')

    if is_generation:
        return {
            'predictions': probs,
        }
    else:
        x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=x_indices, name='nll'),
            0,
            name='loss')
        return {
            'predictions': probs,
            'loss': loss,
            'eval': {
                'nll': loss
            },
            'quantized_input': x_quantized,
            'encoding': lc,
            'global_condition': gc,
        }
