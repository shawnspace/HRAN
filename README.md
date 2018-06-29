# HRAN
The inplementation of Hierarchical Recurrent Attention Network (Xing, Chen, et al. "Hierarchical Recurrent Attention Network for Response Generation." arXiv preprint arXiv:1701.07149 (2017))

You need to prepare your own data files which consists of three text files: dialog_train.txt, dialog_valid.txt, rg_vocab.txt.

The format of dialog_train.txt file and dialog_valid.txt file is:
q1\ta1\tq2\ta2\n
q1\ta1\tq2\ta2\n
...

For the rg_vocab.txt file, it contains all the unique words separated by new line with unk being index 0 and sos being index 1 and eos being index 2.

For each line, there are several utterances (like q1, a1 here) and you can split them by '\t'. For each utterance, I have already conducted word tokenization and you can split each utterance by whitespace to get each word token.

To train the model, you need to firstly use prepare_context_RG_data.py to generate the .tfrecords files. Then you can use train.py to train the model.
