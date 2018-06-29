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
tf.flags.DEFINE_string('model_dir','./model/persona_chat3','model dir')
tf.flags.DEFINE_string('data_dir','/qydata/xzhangax/my_project/data/persona_chat/multi','data dir')
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
        loss,debug_tensors = HRAN.impl(features=input_features,mode=modekeys.TRAIN,hp=hp)
        global_step_tensor = tf.Variable(initial_value=0,
                                         trainable=False,
                                         collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
                                         name='global_step')
        train_op, grad_norm = create_train_op(loss, hp.learning_rate, global_step_tensor, hp.clip_norm)
        stop_criteria_tensor = tf.Variable(initial_value=10000, trainable=False, name='stop_criteria', dtype=tf.float32)

        tf.summary.scalar(name='train_loss',tensor=loss)
        tf.summary.scalar(name='train_grad_norm', tensor=grad_norm)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=os.path.join(os.path.abspath(MODEL_DIR), 'summary'))

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        if FLAGS.debug:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess,thread_name_filter = "MainThread$")
            #sess.add_tensor_filter(tensor_filter=tf_debug.has_inf_or_nan,filter_name='has_inf_or_nan')
            pass

        saver = tf.train.Saver(max_to_keep=1)
        best_saver = tf.train.Saver(max_to_keep=1)
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
            stop_count = 10

            while not coord.should_stop():
                _,current_loss,summary,global_step = sess.run(fetches=[train_op,loss,summary_op,global_step_tensor])

                if global_step % 100 == 0:
                    tf.logging.info('global step '+str(global_step)+' loss: ' + str(current_loss))

                if global_step % hp.summary_save_steps == 0:
                    summary_writer.add_summary(summary=summary,global_step=global_step)

                if global_step % hp.eval_step == 0:
                    saver.save(sess=sess, save_path=os.path.join(MODEL_DIR, 'model.ckpt'), global_step=global_step)
                    eval_file = os.path.join(os.path.abspath(FLAGS.data_dir), 'valid.tfrecords')
                    cur_stop_criteria = evaluate.evaluate(eval_file, MODEL_DIR, os.path.join(MODEL_DIR, 'summary/eval'),
                                                      global_step)
                    stop_criteria = sess.run(stop_criteria_tensor)
                    if cur_stop_criteria < stop_criteria:
                        sess.run(stop_criteria_tensor.assign(cur_stop_criteria))
                        best_model_path = os.path.join(os.path.join(MODEL_DIR, 'best_model'), 'model.ckpt')
                        save_path = best_saver.save(sess=sess, save_path=best_model_path,
                                                    global_step=tf.train.get_global_step())
                        tf.logging.info('Save best model to {}'.format(save_path))
                        stop_count = 10
                    else:
                        stop_count -= 1
                        if stop_count == 0:
                            tf.logging.info('Early stop at step {}'.format(global_step))
                            break


        except tf.errors.OutOfRangeError:
            tf.logging.info('Finish training -- epoch limit reached')
        finally:
            tf.logging.info('Best ppl is {}'.format(sess.run(stop_criteria_tensor)))
            coord.request_stop()

        coord.join(threads)

def create_train_op(loss,lr,global_step,clip_norm):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grad_var = optimizer.compute_gradients(loss)
    # grad_var = [(tf.clip_by_value(grad, clip_value_min=-100, clip_value_max=clip_value_max), var) for grad, var in
    #             grad_var]
    grad_var = [(tf.clip_by_norm(grad, clip_norm=clip_norm), var) for grad, var in grad_var]
    train_op = optimizer.apply_gradients(grads_and_vars=grad_var, global_step=global_step)

    debug_tensors = [gv[0] for gv in grad_var]

    grads = [gv[0] for gv in grad_var]
    grad_norm = tf.global_norm(grads)
    return train_op, grad_norm

if __name__ == '__main__':
    tf.app.run()
