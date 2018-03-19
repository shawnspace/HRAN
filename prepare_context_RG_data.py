import tensorflow as tf
import random
import numpy as np

max_sen_length = 30
max_con_length = 10

def load_vocabulary(vocab_path): #41834 word
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()):
            vocabulary[l.rstrip('\n')] = i
    print(len(vocabulary))
    return vocabulary

def create_dataset():
    vocab = load_vocabulary('./twitter_data/rg_vocab.txt')
    examples = []
    with open('./twitter_data/dialog.txt', 'r') as f:
        for l in f.readlines():
            sentences = l.split('\t')
            response = sentences[-1]
            contexts = sentences[0:len(sentences) - 1]
            example = create_example(contexts,response,vocab)
            examples.append(example)

    random.shuffle(examples)

    valid_examples = examples[0:10000]
    test_examples = examples[10000:11000]
    train_examples = examples[11000:]

    writer = tf.python_io.TFRecordWriter('./twitter_data/train.tfrecords')
    for e in train_examples:
        writer.write(e.SerializeToString())
    print(len(train_examples))

    writer = tf.python_io.TFRecordWriter('./twitter_data/valid.tfrecords')
    for e in valid_examples:
        writer.write(e.SerializeToString())
    print(len(valid_examples))

    writer = tf.python_io.TFRecordWriter('./twitter_data/test.tfrecords')
    for e in test_examples:
        writer.write(e.SerializeToString())
    print(len(test_examples))

    print(len(examples))

def create_example(contexts,response,vocabulary):
    def transform_utterance(query,vocabulary):
        query = query.split(' ')
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

    context_ids = []
    context_utterance_len = []
    for c in contexts:
        c_id, c_len = transform_utterance(c,vocabulary)
        context_ids.append(c_id)
        context_utterance_len.append(c_len)

    context_len = len(context_ids)

    if context_len <= max_con_length:
        dummy_c = [0]*max_sen_length
        context_ids.extend([dummy_c]*(max_con_length - context_len))
        context_utterance_len.extend([1]*(max_con_length - context_len))
    else:
        context_ids = context_ids[(len(context_ids) - max_con_length):len(context_ids)]
        context_len = max_con_length
        context_utterance_len = context_utterance_len[(len(context_utterance_len) - max_con_length):len(context_utterance_len)]

    response_ids = []
    for w in response.split(' '):
        try:
            response_ids.append(vocabulary[w])
        except KeyError:
            response_ids.append(0)

    response_length = len(response_ids)
    if response_length <= max_sen_length-1:
        response_out = response_ids + [1]
        response_mask = [1] * len(response_out)
        response_mask.extend([0]*(max_sen_length-len(response_out)))
        response_out.extend([0]*(max_sen_length-len(response_out)))
        response_in = [1] + response_ids
        response_in.extend([0] * (max_sen_length - len(response_in)))
        response_length += 1
    else:
        response_out = response_ids[0:max_sen_length-1] + [1]
        response_mask = [1] * max_sen_length
        response_in = [1] + response_ids[0:max_sen_length-1]
        response_length = max_sen_length


    context_ids = np.array(context_ids,dtype=np.int64)
    features = {'contexts_flatten': _int64_feature(context_ids.flatten()),
                'context_utterance_length': _int64_feature(context_utterance_len),
                'context_length': _int64_feature(value=[context_len]),
                'response_out': _int64_feature(response_out),
                'response_mask': _int64_feature(response_mask),
                'response_in':_int64_feature(response_in)}

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example

def _int64_feature(value):
    # parameter value is a list
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

create_dataset()