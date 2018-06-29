import tensorflow as tf
from collections import namedtuple
import os

# Model Parameters
tf.flags.DEFINE_string('word_embed_path','/qydata/xzhangax/my_project/data/persona_chat/my_vector.txt','path to word embedding')
tf.flags.DEFINE_string('vocab_path','/qydata/xzhangax/my_project/data/persona_chat/rg_vocab.txt','vocab path')
tf.flags.DEFINE_integer('vocab_size',18423,'vocab size')
tf.flags.DEFINE_integer("word_dim", 300, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer('word_rnn_num_units', 600, 'Num of rnn cells')
tf.flags.DEFINE_integer('context_rnn_num_units', 1200, 'Num of rnn cells')
tf.flags.DEFINE_integer('decoder_rnn_num_units', 1200, 'Num of rnn cells')
tf.flags.DEFINE_integer('context_attn_units',100,'num context attn units')
tf.flags.DEFINE_integer('utte_attn_units',100,'num utterance level attn units')
tf.flags.DEFINE_integer('beam_width', 10, 'Num of beam_width')
tf.flags.DEFINE_float('keep_prob', 1.0, 'the keep prob of rnn state')
tf.flags.DEFINE_string('rnn_cell_type', 'GRU', 'the cell type in rnn')

# Pre-trained parameters
tf.flags.DEFINE_integer('max_sentence_length', 25,'the max sentence length')
tf.flags.DEFINE_integer('max_context_length', 20,'the max context length')

# Training Parameters,
# train example 131438 4000 step /epoch
# valid example 3907
# test example 3894
tf.flags.DEFINE_integer("batch_size", 32, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 64, "Batch size during evaluation")
tf.flags.DEFINE_integer('num_epochs', 10, 'the number of epochs')
tf.flags.DEFINE_integer('eval_step', 2000, 'eval every n steps')
tf.flags.DEFINE_boolean('shuffle_batch',True, 'whether shuffle the train examples when batch')
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer('summary_save_steps',200,'steps to save summary')

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
    'context_attn_units',
    'utte_attn_units',
    'beam_width',
    'keep_prob',
    'rnn_cell_type',
    'max_sentence_length',
    'max_context_length',
    'shuffle_batch',
    'summary_save_steps',
    'clip_norm',
    'lambda_l2',
    'word_embed_path',
    'vocab_path'
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
        context_attn_units = FLAGS.context_attn_units,
        utte_attn_units = FLAGS.utte_attn_units,
        beam_width=FLAGS.beam_width,
        keep_prob=FLAGS.keep_prob,
        rnn_cell_type=FLAGS.rnn_cell_type,
        max_sentence_length=FLAGS.max_sentence_length,
        max_context_length=FLAGS.max_context_length,
        shuffle_batch=FLAGS.shuffle_batch,
        summary_save_steps=FLAGS.summary_save_steps,
        lambda_l2=0.001,
        clip_norm=1000000000000000,
        word_embed_path=FLAGS.word_embed_path,
        vocab_path=FLAGS.vocab_path
    )

def write_hparams_to_file(hp, model_dir):
  with open(os.path.join(os.path.abspath(model_dir),'hyper_parameters.txt'), 'w') as f:
    f.write('batch_size: {}\n'.format(hp.batch_size))
    f.write('learning_rate: {}\n'.format(hp.learning_rate))
    f.write('num_epochs: {}\n'.format(hp.num_epochs))
    f.write('word_rnn_num_units: {}\n'.format(hp.word_rnn_num_units))
    f.write('keep_prob: {}\n'.format(hp.keep_prob))
