# '''
# Solution
# '''
# import numpy as np
# np.random.seed(400)
# import pandas as pd
# # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# df = pd.read_table('/Users/joeDiHare/Documents/ODSCnlp/ODSC/bayesian_inference/smsspamcollection/SMSSpamCollection',
#                    sep='\t',
#                    header=None,
#                    names=['label', 'sms_message'])
#
# # Output # Output printing out first 5 columns
# df.head()
#
# df['label'] = df.label.map({'ham':0, 'spam':1})
# print(df.shape)
# df.head() # returns (rows, columns)
#
# documents = ['Hello, how are you!',
#              'Win money, win from home.',
#              'Call me now.',
#              'Hello, Call hello you tomorrow?']
#
# lower_case_documents = []
# for i in documents:
#     lower_case_documents.append(i.lower())
# print(lower_case_documents)printing out first 5 columns


import tensorflow as tf
import numpy as np
import os
import re

# catch_sys_msg = '^([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}), ([0-9]{2}:[0-9]{2}):?[0-9]{0,2}(:\s|\s\-\s){1}([^:]+)$'  # catches ONLY system messages
# catch_usr_msg = '^([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}), ([0-9]{2}:[0-9]{2}):?[0-9]{0,2}(:\s|\s\-\s){1}:{0,10}([^:]{1,25}):{0,10}(.*)$'  # catches system and user messages
# catch_url = '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'  # carches urls
# catch_media = '(<[a-zA-Z\s]{5,30}>)'
# results, mediaExchange, R, SystemMsg = [], [], [], []
# for filename in ['chat_full.txt']:
#     for row in open(os.path.join('/Users/joeDiHare/Documents/WhatChat/dev-algorithm', filename)):
#         if len(row.strip()):
#             if re.search(catch_sys_msg, row) is None and row != '':
#                 if re.search(catch_usr_msg, row) is not None:
#                     mtch = re.search(catch_usr_msg, row)
#                     # logging.debug('usr msg ~ ' + row)
#                     results.append([mtch.group(1), mtch.group(2), mtch.group(4), mtch.group(5)])
#                 else:
#                     # logging.debug('\nusr msg only text ~ ' + row)
#                     results[-1][-1] = results[-1][-1] + ' ' + row[0]
#                 # search for links, media
#                 if re.search(catch_url, row) is not None:
#                     # logging.debug('link')
#                     mediaExchange.append([mtch.group(1), mtch.group(2), re.search(catch_url, row).group(1)])
#                 if re.search(catch_media, row) is not None:
#                     mediaExchange.append([mtch.group(1), mtch.group(2), re.search(catch_media, row).group(1)])
#                     results[-1][-1] = results[-1][-1].replace(re.search(catch_media, row).group(1), ' ')
#                     # logging.debug('media')
#             elif row != '':
#                 SystemMsg.append(row)
# names = set()
# for n in results:
#     tmp = n[3].strip().split()
#     for n in tmp:
#         tmp =''.join([c for c in n.strip() if c not in '!@£$%^&*()_+{}:|<>?|,.;[]\=-"1234567890`~']).lower()
#         if len(tmp.strip()) and len(tmp.strip())<11 and tmp[:4]!='http':
#             names.add(tmp)

names = set()
for filename in ['male.txt', 'female.txt']:
    for line in open(os.path.join('/Users/joeDiHare/Documents/notebooks/juypter-notebooks', filename)):
        if len(line.strip()):
            names.add(line.strip().lower())

chars = list('abcdefghijklmnopqrstuvwxyz') + ['<END>', '<NULL>']
indices_for_chars = {c: i for i, c in enumerate(chars)}

NAME_MAX_LEN = 10 # include the <END> char

def name_to_vec(name, maxlen=NAME_MAX_LEN):
    v = np.zeros(maxlen, dtype=int)
    null_idx = indices_for_chars['<NULL>']
    v.fill(null_idx)
    for i, c in enumerate(name):
        if i >= maxlen: break
        n = indices_for_chars.get(c, null_idx)
        v[i] = n
    v[min(len(name), maxlen-1)] = indices_for_chars['<END>']
    return v

def vec_to_name(vec):
    name = ''
    for x in vec:
        char = chars[x]
        if len(char) == 1:
            name += char
        elif char == '<END>':
            return name
    return name

print(name_to_vec('nate'))
assert vec_to_name(name_to_vec('nate')) == 'nate'
assert vec_to_name(name_to_vec('aaaaaaaaaaaa')) == 'aaaaaaaaa'

name_vecs = np.array([name_to_vec(n) for n in names])
print(name_vecs.shape)

#

def weight_var(shape, stddev=0.1, weight_decay=0, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    v = tf.Variable(initial, name=name)
    if weight_decay > 0:
        l2 = tf.nn.l2_loss(v) * weight_decay
        tf.add_to_collection('losses', l2)
    return v


def leaky_relu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def relu(x):
    # return tf.nn.relu(x)
    return leaky_relu(x)

# from tensorflow import create_batch_norm

def create_conv(input, out_channels, patch_size=5, stride=1, batch_norm=False, dropout=False):
    in_channels = input.get_shape()[-1].value
    w = weight_var([patch_size, patch_size, in_channels, out_channels])
    b = weight_var([out_channels], stddev=0)
    conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')
    if batch_norm: conv = create_batch_norm(conv)
    activation = relu(conv + b)
    if dropout: activation = create_dropout(activation)
    return activation

def text_conv(input, out_channels, patch_size=5, stride=1, dropout=False, pool_size=1):
    in_channels = input.get_shape()[-1].value
    w = weight_var([patch_size, in_channels, out_channels])
    b = weight_var([out_channels], stddev=0)
    conv = tf.nn.conv1d(input, w, stride=stride, padding='SAME')
    activation = relu(conv + b)
    # TODO: max_pooling
    if dropout: activation = create_dropout(activation)
    return activation


def create_dropout(units):
    return tf.nn.dropout(units, dropout)


def create_fc(input, out_size):
    # input_dropped = tf.nn.dropout(input, dropout_keep_prob)
    in_size = input.get_shape()[-1].value
    w = weight_var([in_size, out_size], weight_decay=0.004)
    b = weight_var([out_size], weight_decay=0.004)
    x = tf.matmul(input, w)
    return relu(x + b)

#
name_placeholder = tf.placeholder(shape=[None, NAME_MAX_LEN], dtype=tf.int32, name='names')

#
Z_SIZE = 64

def encoder_lstm(names):
    with tf.variable_scope('encoder'):
        cells = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in [len(chars), 64]]
        lstm = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        # cells = [tf.contrib.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in [len(chars), 64]]
        # lstm = tf.contrib.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        one_hot = tf.one_hot(names, len(chars), dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(lstm, one_hot, dtype=tf.float32)
        outputs_flat = tf.reshape(outputs, [-1, 64 * NAME_MAX_LEN])
        z_mean = create_fc(outputs_flat, Z_SIZE)
        z_stddev = create_fc(outputs_flat, Z_SIZE)
        return z_mean, z_stddev


def encoder_conv(names):
    with tf.variable_scope('encoder'):
        one_hot = tf.one_hot(names, len(chars), dtype=tf.float32)
        conv1 = text_conv(one_hot, 64)
        conv2 = text_conv(one_hot, 64)
        fc1 = create_fc(tf.reshape(conv2, [-1, NAME_MAX_LEN * 64]), 128)
        z_mean = create_fc(fc1, Z_SIZE)
        z_stddev = create_fc(fc1, Z_SIZE)
        return z_mean, z_stddev


# def generator(noise, name='generator'):
#     with tf.variable_scope(name, reuse=None):
#         cells = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in [NOISE_SIZE, 256, len(chars)]]
#         lstm = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
#         noise_repeated_over_time = tf.tile(tf.reshape(noise, [-1, 1, NOISE_SIZE]), [1, NAME_MAX_LEN, 1])
#         outputs, state = tf.nn.dynamic_rnn(lstm, noise_repeated_over_time, dtype=tf.float32)
#         output_chars = tf.reshape(tf.argmax(tf.nn.softmax(outputs), axis=2), [-1, NAME_MAX_LEN])
#         output_chars = tf.cast(output_chars, tf.int32)
#     return output_chars

# generated_names = generator(noise)

z_mean, z_stddev = encoder_lstm(name_placeholder)

#
session = tf.Session()
session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

#
def sample_z(z_mean, z_stddev):
    samples = tf.random_normal(tf.shape(z_stddev), 0, 1, dtype=tf.float32)
    return z_mean + samples * z_stddev

z_vals = sample_z(z_mean, z_stddev)

def decoder_lstm(z):
    z_repeated_over_time = tf.tile(tf.reshape(z, [-1, 1, Z_SIZE]), [1, NAME_MAX_LEN, 1])
    cells = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True) for size in [Z_SIZE, 256, len(chars)]]
    lstm = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(lstm, z_repeated_over_time, dtype=tf.float32)
    return outputs

z_input = tf.placeholder(tf.float32, [None, Z_SIZE], name='z_input')
use_z_input = tf.placeholder(tf.int32, shape=[], name="use_z_input_condition")
decoder_input = tf.cond(use_z_input > 0, lambda: z_input, lambda: z_vals)

decoded = decoder_lstm(decoder_input)

diff_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(decoded, name_placeholder))
kl_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1))
loss = diff_loss + kl_divergence

decoded_vecs = tf.argmax(decoded, axis=2)

learn_rate = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learn_rate)
global_step = tf.contrib.framework.get_or_create_global_step()
train_step = optimizer.minimize(loss, global_step=global_step)
session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

#

save_path = '/Users/joeDiHare/Documents/notebooks/juypter-notebooks/models/nva2'

session = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
session.run(init_op)

saver = None
if save_path:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print('Restored from checkpoint', ckpt.model_checkpoint_path)
    else:
        print('Did not restore from checkpoint')
else:
    print('Will not save progress')


# Train
train = False
while train:
    names = name_vecs[np.random.randint(name_vecs.shape[0], size=64), :]
    feed_dict = {
        name_placeholder: names,
        z_input: np.zeros((64, Z_SIZE)),
        use_z_input: 0,
        learn_rate: 0.001
    }
    _, loss_, step_ = session.run([train_step, loss, global_step], feed_dict=feed_dict)
    if step_ % 200 == 0:
        output_ = session.run(decoded_vecs, feed_dict=feed_dict)
        print("Step: {0}; loss: {1}".format(step_, loss_))
        # print names[0]
        # print output_[0]
        print(" example encoding: {} -> {}".format(vec_to_name(names[0]), vec_to_name(output_[0])))
        if step_ % 600 == 0:
            saver.save(session, save_path + '/model.ckpt', global_step=step_)
            print('Saved')

def reconstruct(name):
    feed_dict = {
        name_placeholder: np.array([name_to_vec(name)]),
        z_input: np.zeros((64, Z_SIZE)),
        use_z_input: 0,
        learn_rate: 0.01
    }
    output_ = session.run(decoded_vecs, feed_dict=feed_dict)
    return vec_to_name(output_[0])

#

for name in ['nate', 'will', 'chen', 'atty', 'arielle', 'nathaniel', 'kimberly', 'erica', 'zoe']:
    print(name, '->', reconstruct(name))
for name in ['word', 'happy', 'winter', 'candle', 'cherish']:
    print(name, '->', reconstruct(name))
    # it's worse at more longer, more "wordy" names:
for name in ['embedding', 'automobile', 'air', 'larynx']:
    print(name, '->', reconstruct(name))
for name in ['ufhoe', 'xyzy', 'ihwrfoecoei']:
    print(name, '->', reconstruct(name))


def nameliness(word):
    r = reconstruct(word)
    return sum([1 if a == b else 0 for a, b in zip(word, r)]) / float(len(word))

for name in ['nate', 'july', 'fridge', 'gienigoe', 'chzsiucf', 'xyxyzzy']:
    print(name, ':', nameliness(name))

# top_words = list(word.strip() for word in open('../data/google-10000-english.txt'))
# top_words = list(word for word in top_words if word not in names)
# print(len(top_words))
# top_words = top_words[:1000] # this is actually kinda slow, so let's stick with the top 1k
# nameliness_scores = {word: nameliness(word) for word in top_words}
# print([w for w in top_words if nameliness_scores[w] == 1])

# let's build a big lookup table of all the names and their embeddings:
def make_batches(list, size=128):
    batches = []
    while len(list):
        batches.append(list[:min(len(list), size)])
        list = list[len(batches[-1]):]
    return batches


embeddings = {}

for batch in make_batches(list(names)):
    feed_dict = {
        name_placeholder: np.array([name_to_vec(name) for name in batch]),
        z_input: np.zeros((len(batch), Z_SIZE)),
        use_z_input: 0
    }
    output_ = session.run(z_mean, feed_dict=feed_dict)
    for name, vec in list(zip(batch, output_)):
        # print(name,vec)
        embeddings[tuple(name)] = vec

def embed(name):
    feed_dict = {
        name_placeholder: np.array([name_to_vec(name)]),
        z_input: np.zeros((1, Z_SIZE)),
        use_z_input: 0
    }
    output_ = session.run(z_mean, feed_dict=feed_dict)
    return output_[0]

def nearest(embedding):
    def distance(name):
        return np.linalg.norm(embedding - embeddings[name])
    return ''.join(min(embeddings.keys(), key=distance))

def unembed(embedding):
    feed_dict = {
        name_placeholder: np.zeros((1, NAME_MAX_LEN)),
        z_input: np.array([embedding]),
        use_z_input: 1
    }
    output_ = session.run(decoded_vecs, feed_dict=feed_dict)
    return vec_to_name(output_[0])

print(unembed(embed('nate')) == 'nate')

# print nearest(embed('nate'))
for name in ['nate', 'yikes', 'panda', 'ixzhxzi', 'justxn']: print(name, 'is closest to', (nearest(embed(name))))

#
# what happens if we try to interpolate names?
def blend_names(name1, name2):
    e1 = embed(name1)
    e2 = embed(name2)
    for i in range(11):
        blend = i / 10.0
        print(unembed(e1 * (1 - blend) + e2 * blend))

blend_names('amy', 'francisco')
blend_names('nathaniel', 'chen')
blend_names('will', 'william')
print(nearest(np.zeros(Z_SIZE)))
print(unembed(np.zeros(Z_SIZE)))
for name in ['nate', 'willy', 'sam', 'polly', 'jacob']:
    print(name, '* 2 =', unembed(embed(name) * 2))
# what's the opposite of a name?
for name in ['nancy', 'barry', 'chance', 'rachel', 'gloria']:
    print('-' + name, '=', unembed(-embed(name)))
# can we do addition and subtraction?
print(unembed(embed('alberta') - embed('albert') + embed('robert')))
print(unembed(embed('alberta') - embed('albert') + embed('justin')))
print(unembed(embed('alberta') - embed('albert') + embed('joseph')))
print(unembed(embed('alberta') - embed('albert') + embed('nate'))) # doesn't work so well with names ending in vowels
#
# let's generate some random names:
def generate():
    return unembed(np.random.normal(size=Z_SIZE))
for _ in range(10):
    print(generate())
#

def variations_on(name):
    z = embed(name)
    for i in range(10):
        noise = np.random.normal(z.shape)
        print(unembed(z + noise * i * 0.01))

variations_on('nate')
