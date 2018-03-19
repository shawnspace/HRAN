import tensorflow as tf
import modekeys
import hparam
import os
import HRAN
from tensorflow.python.training import saver as saver_lib
from tensorflow.python import debug as tf_debug
import evaluate
import datetime

#use dropout
#use gradient clip
# add momentum
# penalize unk words
# truncated BPTT

def main(unused_arg):
    tf.logging.set_verbosity(tf.logging.INFO)
    train()

tf.flags.DEFINE_boolean('debug',False,'debug mode')
tf.flags.DEFINE_string('model_dir','./model/twitter_model3','model dir')
tf.flags.DEFINE_string('data_dir','./twitter_data','data dir')
FLAGS = tf.flags.FLAGS

TRAIN_FILE = os.path.join(os.path.abspath(FLAGS.data_dir), 'train.tfrecords')
MODEL_DIR = FLAGS.model_dir
if MODEL_DIR is None:
    timestamp = datetime.datetime.now()
    MODEL_DIR = os.path.join(os.path.abspath('./model'), str(timestamp))

def train():
    hp = hparam.create_hparam()
    train_graph = tf.Graph()
    with train_graph.as_default():
        input_features = HRAN.create_input_layer(mode=modekeys.TRAIN,filename=TRAIN_FILE,hp=hp)
        loss = HRAN.impl(features=input_features,mode=modekeys.TRAIN,hp=hp)
        global_step_tensor = tf.Variable(initial_value=0,
                                         trainable=False,
                                         collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
                                         name='global_step')
        train_op,lr = create_train_op(loss,hp.learning_rate,global_step_tensor)

        tf.summary.scalar(name='train_loss',tensor=loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=os.path.join(os.path.abspath(MODEL_DIR), 'summary'))

        sess = tf.Session()

        if FLAGS.debug:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess,thread_name_filter = "MainThread$")
            #sess.add_tensor_filter(tensor_filter=tf_debug.has_inf_or_nan,filter_name='has_inf_or_nan')
            pass

        saver = tf.train.Saver(max_to_keep=5)
        checkpoint = saver_lib.latest_checkpoint(MODEL_DIR)
        tf.logging.info('model dir {}'.format(MODEL_DIR))
        tf.logging.info('check point {}'.format(checkpoint))
        if checkpoint:
            tf.logging.info('Restore parameter from {}'.format(checkpoint))
            saver.restore(sess=sess,save_path=checkpoint)
            sess.run(tf.local_variables_initializer())
        else:
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.logging.info(msg='Begin training')
        try:
            while not coord.should_stop():
                _,current_loss,summary,global_step,learning_rate = sess.run(fetches=[train_op,loss,summary_op,global_step_tensor,lr])

                if global_step % 100 == 0:
                    tf.logging.info('global step '+str(global_step)+' loss: ' + str(current_loss))

                if global_step % hp.summary_save_steps == 0:
                    summary_writer.add_summary(summary=summary,global_step=global_step)
                    tf.logging.info('learning rate {}'.format(learning_rate))

                if global_step % hp.eval_step == 0:
                    tf.logging.info('save model')
                    saver.save(sess=sess,save_path=os.path.join(MODEL_DIR, 'model.ckpt'),global_step=global_step)
                    eval_file = os.path.join(os.path.abspath(FLAGS.data_dir), 'valid.tfrecords')
                    evaluate.evaluate(eval_file,MODEL_DIR,os.path.join(MODEL_DIR, 'summary/eval'),global_step)

        except tf.errors.OutOfRangeError:
            tf.logging.info('Finish training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

        saver.save(sess=sess,save_path=os.path.join(MODEL_DIR, 'model.ckpt'),global_step=tf.train.get_global_step())

def create_train_op(loss,lr,global_step_tensor):
    optimizer = tf.train.AdamOptimizer(lr)
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    #clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    train_op = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),global_step=global_step_tensor)
    return train_op,optimizer._lr_t

if __name__ == '__main__':
    tf.app.run()
