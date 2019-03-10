from __future__ import division

from flask import render_template, request

from app import app
from chatbot.config import basedir

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

import numpy as np
import pandas as pd
import re
import os

import pickle

CONTRACTIONS = {
    "i ain't": "am not",
    "you ain't": "you are not",
    "they ain't": "they are not",
    "she ain't": "she is not",
    "he ain't": "he is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
CONTRACTIONS_RE = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))
NAME_LIST = pd.read_csv(os.path.join(basedir, "data/firstnames.csv"))["firstname"].tolist()

def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 separated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs=encode_seqs,
                vocabulary_size=xvocab_size,
                embedding_size=emb_dim,
                name='seq_embedding')

            vs.reuse_variables()
            tl.layers.set_name_reuse(True)

            net_decode = EmbeddingInputlayer(
                inputs=decode_seqs,
                vocabulary_size=xvocab_size,
                embedding_size=emb_dim,
                name='seq_embedding')

        net_rnn = Seq2Seq(net_encode, net_decode,
                          cell_fn=tf.contrib.rnn.BasicLSTMCell,
                          n_hidden=emb_dim,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length=retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode=None,
                          dropout=(0.5 if is_train else None),
                          n_layer=3,
                          return_seq_2d=True,
                          name='seq2seq')

        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
    return net_out, net_rnn

with open(os.path.join(basedir, r'models/metadata_punc.pkl'), 'rb') as f:
    metadata = pickle.load(f)

xvocab_size = metadata["xvocab_size"]
w2idx = metadata["w2idx"]
idx2w = metadata["idx2w"]
emb_dim = metadata["emb_dim"]

start_id = w2idx["start_id"]
end_id = w2idx["end_id"]

# model for training
batch_size = 16

encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") # tl.prepro.sequences_get_mask()
net_out, _ = model(encode_seqs, decode_seqs, is_train=True, reuse=False)

# model for inferencing
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
net, net_rnn = model(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
y = tf.nn.softmax(net.outputs)

os.chdir(os.path.join(basedir, r'models'))

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name='n_withPunc.npz', network=net)


def expand_contractions(s, contractions_dict=CONTRACTIONS, contractions_re=CONTRACTIONS_RE):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def clean_text(text, name_list=NAME_LIST):
    # get rid of common english names
    text = " ".join([w for w in text.split(" ") if re.sub(r'[^\w\d_\s]+', '', w) not in name_list])
     
    # to lower case
    text = text.lower()
    # remove @ mentions
    text = re.sub(r'@\w+\b', '', text)
    # remove url links
    text = re.sub(r'\bhttp.+\b', '', text)
    # remove line break
    text = re.sub(r'\n', ' ', text)
    
    # expand contraction
    text = expand_contractions(text)
    
    # replace some common shorthands
    text = re.sub(r'&amp;', ' and ', text)
    text = re.sub(r'\bb/c\b', 'because', text)
    text = re.sub(r'\b&lt;\b', '<', text)
    text = re.sub(r'\b&gt;\b', '>', text)
    
    # remove punctuation
    # text = re.sub(r'[^\w\d_\s]+', '', text)
    
    # get rid of initials at the end
    if len(text) > 0:
        if re.match(r'(?:\/|\^|\*)[A-z]{2}', text[-1]) is not None:
            text.pop()
        
    if len(text) <= 3:
        return "TO_DELETE"
    else:
        return " ".join(text.split())


# landing page
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# get response
@app.route('/get_response', methods=['POST'])
def get_response():

    seed = str(request.json["input"].replace(u'\u2019', "'").replace('"', "").replace("&#39;", '"'))
    print("INPUT: " + seed)

    seed = clean_text(seed)
    print("CLEANED INPUT: " + seed)

    seed_id = [w2idx[w] for w in seed.split(" ") if w in w2idx.keys()]

    response_list = []

    for _ in range(1):  # 1 Query --> 1 Reply
        #  1. encode, get state
        state = sess.run(net_rnn.final_state_encode, {encode_seqs2: [seed_id]})
        # 2. decode, feed start_id, get first word
        o, state = sess.run([y, net_rnn.final_state_decode], {net_rnn.initial_state_decode: state, 
                                                              decode_seqs2: [[start_id]]})
        w_id = tl.nlp.sample_top(o[0], top_k=3)
        w = idx2w[w_id]
        
        # sort and save probabilities
        probs = []
        probabilities = o[0][w_id]
        
        # this stores the probability of the top word each time
        probs = np.append(probs, probabilities)
        
        # 3. decode, feed state iteratively
        if w != "unk":
            sentence = [w]
        else:
            sentence = []

        for _ in range(500): # max sentence length
            o, state = sess.run([y, net_rnn.final_state_decode], {net_rnn.initial_state_decode: state,
                                                                  decode_seqs2: [[w_id]]})
            w_id = tl.nlp.sample_top(o[0], top_k=2)
            w = idx2w[w_id]

            probabilities = o[0][w_id]
                        
            if w_id == end_id:
                break

            if w != "unk":
                sentence = sentence + [w]

            probs = np.append(probs, probabilities)

    print("RESPONSE: " + ' '.join(sentence))
    print(probs)

    return ' '.join(sentence)



