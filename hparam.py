import tensorflow as tf
from collections import namedtuple
import os

# Model Parameters
tf.flags.DEFINE_integer('vocab_size',41834,'vocab size')
tf.flags.DEFINE_integer("word_dim", 200, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer('word_rnn_num_units', 512, 'Num of rnn cells')
tf.flags.DEFINE_integer('context_rnn_num_units', 1024, 'Num of rnn cells')
tf.flags.DEFINE_integer('decoder_rnn_num_units', 1024, 'Num of rnn cells')
tf.flags.DEFINE_integer('beam_width', 5, 'Num of beam_width')
tf.flags.DEFINE_float('keep_prob', 1.0, 'the keep prob of rnn state')
tf.flags.DEFINE_string('rnn_cell_type', 'GRU', 'the cell type in rnn')

# Pre-trained parameters
tf.flags.DEFINE_integer('max_sentence_length', 30,'the max sentence length')
tf.flags.DEFINE_integer('max_context_length', 10,'the max context length')

# Training Parameters, 840963 10000 1000
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 64, "Batch size during evaluation")
tf.flags.DEFINE_integer('num_epochs', 3, 'the number of epochs')
tf.flags.DEFINE_integer('eval_step', 13000, 'eval every n steps')
tf.flags.DEFINE_boolean('shuffle_batch',True, 'whether shuffle the train examples when batch')
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.flags.DEFINE_integer('summary_save_steps',250,'steps to save summary')

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [ "eval_step",
    "batch_size",
    "word_dim",
    "eval_batch_size",
    "learning_rate",
    'vocab_size',
    "num_epochs",
    'word_rnn_num_units',
    'context_rnn_num_units',
    'decoder_rnn_num_units',
    'beam_width',
    'keep_prob',
    'rnn_cell_type',
    'max_sentence_length',
    'max_context_length',
    'shuffle_batch',
    'summary_save_steps'
  ])

def create_hparam():
    return HParams(
        eval_step=FLAGS.eval_step,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        learning_rate=FLAGS.learning_rate,
        word_dim=FLAGS.word_dim,
        vocab_size=FLAGS.vocab_size,
        num_epochs=FLAGS.num_epochs,
        word_rnn_num_units=FLAGS.word_rnn_num_units,
        context_rnn_num_units=FLAGS.context_rnn_num_units,
        decoder_rnn_num_units=FLAGS.decoder_rnn_num_units,
        beam_width=FLAGS.beam_width,
        keep_prob=FLAGS.keep_prob,
        rnn_cell_type=FLAGS.rnn_cell_type,
        max_sentence_length=FLAGS.max_sentence_length,
        max_context_length=FLAGS.max_context_length,
        shuffle_batch=FLAGS.shuffle_batch,
        summary_save_steps=FLAGS.summary_save_steps
    )

def write_hparams_to_file(hp, model_dir):
  with open(os.path.join(os.path.abspath(model_dir),'hyper_parameters.txt'), 'w') as f:
    f.write('batch_size: {}\n'.format(hp.batch_size))
    f.write('learning_rate: {}\n'.format(hp.learning_rate))
    f.write('num_epochs: {}\n'.format(hp.num_epochs))
    f.write('word_rnn_num_units: {}\n'.format(hp.word_rnn_num_units))
    f.write('keep_prob: {}\n'.format(hp.keep_prob))
