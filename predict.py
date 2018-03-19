import os
import hparam
import tensorflow as tf
import HRAN
import modekeys
from tensorflow.python.training import saver as saver_lib

def predict(datafile,model_dir):
    hp = hparam.create_hparam()

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        input_features = HRAN.create_input_layer(mode=modekeys.PREDICT, filename=datafile,hp=hp)
        contexts = input_features['contexts']
        response_out = input_features['response_out']
        context_length = input_features['context_length']
        context_utterance_length = input_features['context_utterance_length']
        sample_ids, final_lengths = HRAN.impl(features=input_features, hp=hp, mode=modekeys.PREDICT)
        sess = tf.Session()

        saver = tf.train.Saver()
        checkpoint = saver_lib.latest_checkpoint(model_dir)
        saver.restore(sess=sess, save_path=checkpoint)
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.logging.info('Begin prediction at model {} on file {}'.format(checkpoint, datafile))

        try:
            while not coord.should_stop():
                contexts_ids, gen_ids, gen_lengths, ref_ids,con_len,con_utte_len = sess.run(fetches=[contexts,sample_ids,final_lengths,response_out,context_length,context_utterance_length])
                tf.logging.info('write prediction to file')
                write_to_file(contexts_ids,ref_ids,gen_ids,'./twitter_data',model_dir,gen_lengths,con_len,con_utte_len)
                coord.request_stop()

        except tf.errors.OutOfRangeError:
            tf.logging.info('Finish prediction')
        finally:
            coord.request_stop()
        coord.join(threads)



def write_to_file(context,response,generations,data_dir,model_dir,gen_length,con_len,con_utte_len):
    vocabulary = load_vocabulary(os.path.join(data_dir,'rg_vocab.txt'))
    filepath = os.path.join(model_dir,'generate_response.txt')
    print('gen_length {}'.format(gen_length.shape))
    print('con_len {}'.format(con_len.shape))
    print('con_utte_len {}'.format(con_utte_len.shape))
    with open(filepath,'w') as f:
        for c,r,gen,gen_len,c_len,c_u_len in zip(context,response,generations,gen_length,con_len,con_utte_len):
            if len(set(gen)) >0 :
                c_words = replace_con_to_words(c,vocabulary,c_len,c_u_len)
                r_words = replace_to_words(r,vocabulary,None)
                gen_words = replace_to_words(gen,vocabulary,gen_len)

                for u in c_words:
                    f.write('context:\t'+' '.join(u) + '|||\n')
                f.write('response:\t'+' '.join(r_words) + '|||\n')
                f.write('generation:\t'+' '.join(gen_words) + '|||\n')
                f.write('=======END=======\n')

def replace_to_words(ids,vocab,length):
    if length:
        result = []
        i = 0
        while i < length:
            result.append(vocab[ids[i]])
            i += 1
        return result
    else:
        result = []
        for i in ids:
            result.append(vocab[i])
            if i == 1:
                break
        return result

def replace_con_to_words(cons,vocab,con_len,con_utte_len):
    result = []
    for u,l in zip(cons[0:con_len],con_utte_len[0:con_len]):
        sentence = [vocab[i] for i in u[0:l]]
        result.append(sentence)
    return result


def load_vocabulary(vocab_path):
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()): # unk index = 0 eos index = 1
            vocabulary[i] = l.rstrip('\n')
    return vocabulary

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    predict('./twitter_data/train.tfrecords','./model/twitter_model2')

