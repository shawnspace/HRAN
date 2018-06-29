import tensorflow as tf
import modekeys
from tensorflow.python.training import saver as saver_lib
import nltk
from nltk.chunk import tree2conlltags
import string
import re
import HRAN as model_impl
import hparam
import numpy as np

tf.flags.DEFINE_string('model_dir','model/persona_chat1/best_model','model_dir')
tf.flags.DEFINE_string('dialog_mode','multi','single or multi')

MODEL_DIR = tf.flags.FLAGS.model_dir

def online_prediction():
    hp = hparam.create_hparam()

    vocab_path = hp.vocab_path
    vocab = load_vocabulary(vocab_path)
    reverse_vocab = load_reverse_vocabulary(vocab_path)

    features = model_impl.create_input_layer(filename=None,hp=hp,mode=modekeys.PREDICT)
    results = model_impl.impl(features,modekeys.PREDICT,hp)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver()
    checkpoint = saver_lib.latest_checkpoint(MODEL_DIR)
    if checkpoint:
        saver.restore(sess=sess, save_path=checkpoint)
        print('restore from {}'.format(checkpoint))
        sess.run(tf.local_variables_initializer())
    else:
        raise Exception('no check point')

    print('Finish model initializing')
    raw_query = input('Please enter query\n')


    while raw_query != 'q':

        feed_dict = {}
        if tf.flags.FLAGS.dialog_mode == 'single':
            query_id, query_len = preprocess_raw_query(raw_query, vocab, hp.max_sentence_length)
            feed_dict = {features['utterance']: [query_id], features['utterance_length']: [query_len]}
            # print(new_query)
            # print(query_id)
        elif tf.flags.FLAGS.dialog_mode == 'multi':
            context_ids, context_len, context_utterance_lens = preprocess_raw_context(raw_query,vocab,hp.max_sentence_length,hp.max_context_length)
            feed_dict = {features['contexts']:[context_ids],features['context_utterance_length']:[context_utterance_lens],features['context_length']:[context_len]}

        fetch_dict = {}
        if tf.flags.FLAGS.dialog_mode == 'single':
            if hp.beam_width == 0:
                fetch_dict['response_ids'] = results['response_ids']
                fetch_dict['response_lens'] = results['response_lens']
                fetch_dict['alignment_history'] = results['alignment_history']
                fetch_dict['keywords_prob'] = results['keywords_prob']
            else:

                fetch_dict['response_ids'] = results['response_ids']
                fetch_dict['response_lens'] = tf.constant(0)
                fetch_dict['alignment_history'] = tf.constant(0)
                fetch_dict['keywords_prob'] = results['keywords_prob']
        elif tf.flags.FLAGS.dialog_mode == 'multi':
            if hp.beam_width == 0:
                fetch_dict['response_ids'] = results['response_ids']
                fetch_dict['response_lens'] = results['response_lens']
            else:
                fetch_dict['response_ids'] = results['response_ids']
                fetch_dict['response_lens'] = tf.constant(0)

        fetches  = sess.run(fetches=fetch_dict,feed_dict=feed_dict)

        if tf.flags.FLAGS.dialog_mode == 'single':
            if hp.beam_width > 0:
                gen_responses_ids = fetches['response_ids']
                responses = postprocess_k_generated_response(gen_responses_ids,reverse_vocab)

                response = responses[0]

                for res in response:
                    print('Response: {}'.format(res))
            else:
                # print(gen_responses_ids)
                # print(lens)
                gen_responses_ids = fetches['response_ids']
                lens = fetches['response_lens']
                responses = postprocess_generated_response(gen_responses_ids, lens, reverse_vocab)
                response = responses[0]
                print('Response: {}'.format(response))

                alignment_history = fetches['alignment_history']
                alignment_his = alignment_history[0]

                for i,ali in enumerate(alignment_his[0:lens[0]]):
                    print('word{} {}'.format(i,np.argsort(ali)[::-1][0:5]))
                    print(ali)
                    print('\n')

                key_prob = fetches['keywords_prob']
                print('\n')
                print('keywords prediction')
                print(key_prob[0])
                print(np.argsort(key_prob[0])[::-1][0:5])
        elif tf.flags.FLAGS.dialog_mode == 'multi':
            if hp.beam_width > 0:
                gen_responses_ids = fetches['response_ids']
                responses = postprocess_k_generated_response(gen_responses_ids,reverse_vocab)

                response = responses[0]

                for res in response:
                    print('Response: {}'.format(res))
            else:
                # print(gen_responses_ids)
                # print(lens)
                gen_responses_ids = fetches['response_ids']
                lens = fetches['response_lens']
                responses = postprocess_generated_response(gen_responses_ids, lens, reverse_vocab)
                response = responses[0]
                print('Response: {}'.format(response))

        raw_query = input('Please enter query\n')



def load_vocabulary(vocab_path): #41834 word
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()):
            vocabulary[l.rstrip('\n')] = i
    print(len(vocabulary))
    return vocabulary

def load_reverse_vocabulary(vocab_path):
    vocab = {}
    idx = 0
    with open(vocab_path,'r') as f:
        for l in f.readlines():
            vocab[idx] = l.rstrip('\n')
            idx += 1
    return vocab

def twitter_tokenization(text):
    wnl = nltk.WordNetLemmatizer()
    punctuation = set(string.punctuation)
    punctuation.update(['\'\'', '``', '’', 's', '“', '”', '—'])
    text = re.sub(r'@\w+ ','',text)
    text = re.sub('ain\'t','am not',text)
    text = re.sub('can\'t','can not',text)
    text = re.sub('won\'t', 'will not',text)
    tokens = [w.lower() for w in nltk.word_tokenize(text) if w]
    for i,w in enumerate(tokens):
        if w == 'n\'t':
            tokens[i] = 'not'
        else:
            tokens[i] = wnl.lemmatize(w)
    tokens = [w for w in tokens if w not in punctuation]
    return tokens

def preprocess_raw_query(query,vocabulary,max_sen_length):
    query = twitter_tokenization(query)
    query_ids = []
    for w in query:
        try:
            query_ids.append(vocabulary[w])
        except KeyError:
            query_ids.append(0)
    query_length = len(query_ids)

    if query_length <= max_sen_length:
        query_ids.extend([0] * (max_sen_length - query_length))
    else:
        query_ids = query_ids[0:max_sen_length]
        query_length = max_sen_length

    return query_ids,query_length

def personachat_tokenization(text):
    # nltk word tokenization + nltk NER + lower case
    tokens = [w.lower() for w in nltk.word_tokenize(text)]
    tagged_tokens = []
    for w, pos_tag in nltk.pos_tag(tokens):
        if pos_tag == 'CD':
            tagged_tokens.append(('<number>', 'CD'))
        else:
            tagged_tokens.append((w, pos_tag))

    ne_tree = nltk.ne_chunk(tagged_tokens)
    ne_tag_tokens = tree2conlltags(ne_tree)
    tokens = []
    for w, pos, ne_tag in ne_tag_tokens:
        if ne_tag == 'O':
            tokens.append(w)
        elif ne_tag == 'B-PERSON':
            tokens.append('<person>')
        elif ne_tag == 'I-PERSON':
            continue
        elif ne_tag == 'B-PERCENT':
            tokens.append('<number>')
        elif ne_tag == 'I-PERCENT':
            continue
        else:
            tokens.append(w)
    return tokens

def preprocess_raw_context(contexts,vocabulary,max_sen_len,max_con_len):
    contexts = contexts.split('\t')
    context_ids = []
    context_utterance_lens = []
    for u in contexts:
        tokens = personachat_tokenization(u)
        token_ids = []
        for w in tokens:
            try:
                token_ids.append(vocabulary[w])
            except KeyError:
                token_ids.append(0)

        u_len = len(token_ids)
        if u_len <= max_sen_len:
            token_ids.extend([0]*(max_sen_len - u_len))
        else:
            token_ids = token_ids[0:max_sen_len]
            u_len = max_sen_len
        context_ids.append(token_ids)
        context_utterance_lens.append(u_len)

    context_len = len(context_ids)
    if context_len <= max_con_len:
        dummy_c = [0] * max_sen_len
        context_ids.extend([dummy_c] * (max_con_len - context_len))
        context_utterance_lens.extend([1] * (max_con_len - context_len))
    else:
        context_ids = context_ids[(context_len - max_con_len):context_len]
        context_len = max_con_len
        context_utterance_lens = context_utterance_lens[
                                (len(context_utterance_lens) - max_con_len):len(context_utterance_lens)]

    return context_ids, context_len,context_utterance_lens


def postprocess_generated_response(response_ids,len,reverse_vocab):
    responses = []
    for response_id,l in zip(response_ids,len):
        response = [reverse_vocab[idx] for idx in response_id]
        response = response[0:l]
        responses.append(' '.join(response))

    return responses

def postprocess_k_generated_response(response_ids,reverse_vocab):
    responses = []
    for k_response_id in response_ids:
        k_res = []
        for response_id in k_response_id:
            response = []
            for idx in response_id:
                if idx == 2:
                    break
                response.append(reverse_vocab[idx])
            k_res.append(' '.join(response))
        responses.append(k_res)

    return responses

if __name__ == "__main__":
    online_prediction()