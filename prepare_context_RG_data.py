import tensorflow as tf
import numpy as np
import random

max_sen_length = 25
max_con_length = 20

def load_vocabulary(vocab_path): #18423 word
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()):
            vocabulary[l.rstrip('\n')] = i
    return vocabulary

def create_dataset():
    vocab = load_vocabulary('rg_vocab.txt')

    train_examples = []
    count = 0
    with open('multi/train.txt') as f:
        for l in f.readlines():
            dialog = l.rstrip('\n').split('\t')
            assert len(dialog) % 2 == 0
            for i in range(1,len(dialog),2):
                context = dialog[0:i]
                response = dialog[i]
                e = create_example(context,response,vocab)
                train_examples.append(e)
                count += 1
                if count % 10000 == 0:
                    print(count)

    with open('multi/valid.txt') as f:
        dialogs = [l.rstrip('\n').split('\t') for l in f.readlines()]
    assert len(dialogs) == 1000
    valid_dialogs = dialogs[0:500]
    test_dialogs = dialogs[500:]
    del dialogs

    valid_examples = []
    for dialog in valid_dialogs:
        assert len(dialog) % 2 == 0
        for i in range(1,len(dialog),2):
            context = dialog[0:i]
            response = dialog[i]
            e = create_example(context,response,vocab)
            valid_examples.append(e)
            count += 1
            if count % 1000 == 0:
                print(count)

    test_examples = []
    for dialog in test_dialogs:
        assert len(dialog) % 2 == 0
        for i in range(1, len(dialog), 2):
            context = dialog[0:i]
            response = dialog[i]
            e = create_example(context, response, vocab)
            test_examples.append(e)
            count += 1
            if count % 1000 == 0:
                print(count)

    random.shuffle(train_examples)
    writer = tf.python_io.TFRecordWriter('multi/train.tfrecords')
    for e in train_examples:
        writer.write(e.SerializeToString())
    print('train example {}'.format(len(train_examples)))

    random.shuffle(valid_examples)
    writer = tf.python_io.TFRecordWriter('multi/valid.tfrecords')
    for e in valid_examples:
        writer.write(e.SerializeToString())
    print('valid example {}'.format(len(valid_examples)))

    random.shuffle(test_examples)
    writer = tf.python_io.TFRecordWriter('multi/test.tfrecords')
    for e in test_examples:
        writer.write(e.SerializeToString())
    print('test example {}'.format(len(test_examples)))


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
        response_out = response_ids + [2]
        response_mask = [1] * len(response_out)
        response_mask.extend([0]*(max_sen_length-len(response_out)))
        response_out.extend([0]*(max_sen_length-len(response_out)))
        response_in = [1] + response_ids
        response_in.extend([0] * (max_sen_length - len(response_in)))
        response_length += 1
    else:
        response_out = response_ids[0:max_sen_length-1] + [2]
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

