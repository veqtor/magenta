
r"""The training script that runs the party.

This script requires tensorflow 1.1.0-rc1 or beyond.
As of 04/05/17 this requires installing tensorflow from source,
(https://github.com/tensorflow/tensorflow/releases)

"""

import os
import sys

# internal imports
import numpy as np
import tensorflow as tf

import librosa

from magenta.models.nsynth import utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master", "",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("config", "h512_bo16", "Model configuration name")
tf.app.flags.DEFINE_string("logdir", "/tmp/nsynth",
                           "The log directory for this experiment.")
tf.app.flags.DEFINE_string("train_path", "", "The path to the train tfrecord.")
tf.app.flags.DEFINE_string("wav_out_path", "", "The wave output path.")
tf.app.flags.DEFINE_string("temperature", "0.9999", "temperature.")
tf.app.flags.DEFINE_string("embedding_path", "", "Embedding to use for generation.")
tf.app.flags.DEFINE_string("checkpoint_path", "",
                           "A path to the checkpoint. If not given, the latest "
                           "checkpoint in `expdir` will be used.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")

def write_wav(waveform, filename, sample_rate=16000):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

def main(unused_argv=None):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.config is None:
    raise RuntimeError("No config name specified.")

  config = utils.get_module("wavenet." + FLAGS.config).Config()

  if FLAGS.checkpoint_path:
    checkpoint_path = FLAGS.checkpoint_path
  else:
    expdir = FLAGS.expdir
    tf.logging.info("Will load latest checkpoint from %s.", expdir)
    while not tf.gfile.Exists(expdir):
      tf.logging.fatal("\tExperiment save dir '%s' does not exist!", expdir)
      sys.exit(1)

    try:
      checkpoint_path = tf.train.latest_checkpoint(expdir)
    except tf.errors.NotFoundError:
      tf.logging.fatal("There was a problem determining the latest checkpoint.")
      sys.exit(1)


    if not tf.train.checkpoint_exists(checkpoint_path):
        tf.logging.fatal("Invalid checkpoint path: %s", checkpoint_path)
        sys.exit(1)

    tf.logging.info("Will restore from checkpoint: %s", checkpoint_path)

    with tf.Graph().as_default(), tf.device("/gpu:0"):
        sample_length = FLAGS.sample_length

    wav_placeholder = tf.placeholder(
        tf.float32, shape=[6144, 1])
    lc_placeholder = tf.placeholder(
        tf.float32, shape=[6144 / config.ae_hop_length, config.ae_bottleneck_width])
    gc_placeholder = tf.placeholder(
        tf.float32, shape=[config.gc_bottleneck_width])
    graph = config.build({"wav": wav_placeholder, "en": lc_placeholder, "gc": gc_placeholder}, is_training=False)

    embedding = np.loads(FLAGS.embedding_path)
    embedding = embedding.reshape(-1, embedding.shape[-1])
    sample_length = embedding.shape[0] * config.ae_hop_length
    predictions = graph["predictions"]
    out = (np.random.randn(6144) * 2) - 1
    gc = np.random.randn(config.gc_bottleneck_width)

    samples = tf.placeholder(tf.int32)
    decode = utils.inv_mu_law(samples)

    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    variables_to_restore = ema.variables_to_restore()

    # Create a saver, which is used to restore the parameters from checkpoints
    saver = tf.train.Saver(variables_to_restore)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    # Set the opt_level to prevent py_funcs from being executed multiple times.
    session_config.graph_options.optimizer_options.opt_level = 2
    sess = tf.Session("", config=session_config)

    tf.logging.info("\tRestoring from checkpoint.")
    saver.restore(sess, checkpoint_path)

    for i in range(sample_length):
        embedding_step = i / config.ae_hop_length
        pred = sess.run(
            predictions,
            feed_dict={
                wav_placeholder: out[:-6144],
                gc_placeholder: gc,
                lc_placeholder: embedding[:embedding_step]
            }
        )
        # Scale prediction distribution using temperature.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(pred) / FLAGS.temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged after
        # scaling.
        if FLAGS.temperature == 1.0:
            np.testing.assert_allclose(
                pred, scaled_prediction, atol=1e-5,
                err_msg='Prediction scaling at temperature=1.0 '
                        'is not working as intended.')

        sample = np.random.choice(
            np.arange(256), p=scaled_prediction)
        out.append(sample)
        if i+1 % 2000 == 0:
            decoded = sess.run(decode, feed_dict={samples: out})
            write_wav(decoded, FLAGS.wav_out_path)

    decoded = sess.run(decode, feed_dict={samples: out})
    write_wav(decoded, FLAGS.wav_out_path)

if __name__ == '__main__':
    main()





