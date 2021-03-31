import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

# To support dynamic imports of weekly data (npy, ...) from script dir.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(SCRIPT_DIR)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
from keras_utils import reset_tf_session
import grading
import grading_utils

# # Download weekly resources.
# import download_utils
# download_utils.download_week_6_resources("../readonly/week6")
# download_utils.link_week_6_resources()

# token expires every 30 min
COURSERA_TOKEN = '47iT320zXoc45hqX'
COURSERA_EMAIL = 'yisyang@gmail.com'
SUBMIT = False

grader = grading.Grader(assignment_key="NEDBg6CgEee8nQ6uE8a7OA",
                        all_parts=["19Wpv", "uJh73", "yiJkt", "rbpnH", "E2OIL", "YJR7z"])

L = keras.layers
K = keras.backend
CHECKPOINT_ROOT = ""
IMG_SIZE = 299

# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"
PAD_IDX = None

IMG_EMBED_SIZE = None
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
MAX_LEN = 20  # truncate long captions to speed up training

tf.compat.v1.disable_eager_execution()  # Needed for tf1 placeholder
tf.compat.v1.experimental.output_all_intermediates(True)
s = reset_tf_session()


def main():
    # part1()
    # part2()
    part3()

    if SUBMIT:
        grader.set_answer("19Wpv", [8769, 8769, 1])
        grader.set_answer("uJh73", [1, 1, 8766, 8768, 1, 23303])
        grader.set_answer("yiJkt", [1, 2, 3, 4, 5, -1, 1, 2, 4, 5, 1, 2, 3, 4, 5, -1])
        grader.set_answer("rbpnH", [32, 300, 32, 19, 100, 608, 300, 608, 8769, 608, 608])
        grader.set_answer("E2OIL", 9.077761)
        grader.set_answer("YJR7z", 2.5311574)
        grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

        print('Submitted.')
    else:
        print('SUBMIT variable is off. Done.')


def part1():
    train_captions_indexed, val_captions_indexed, vocab = get_captions_indexed()

    ## GRADED PART, DO NOT CHANGE!
    ans1 = grading_utils.test_vocab(vocab, PAD, UNK, START, END)
    ans2 = grading_utils.test_captions_indexing(train_captions_indexed, vocab, UNK)
    ans3 = grading_utils.test_captions_batching(batch_captions_to_matrix)

    print(ans1)
    print(ans2)
    print(ans3)

    # Vocabulary creation
    grader.set_answer("19Wpv", ans1)
    # Captions indexing
    grader.set_answer("uJh73", ans2)
    # Captions batching
    grader.set_answer("yiJkt", ans3)

    return


def part2():
    s = reset_tf_session()

    img_embed_size = update_img_embed_size()
    decoder = DecoderV1(img_embed_size)
    s.run(tf.compat.v1.global_variables_initializer())
    ans4 = grading_utils.test_decoder_shapes(decoder, img_embed_size, decoder.vocab, s)
    ans5 = grading_utils.test_random_decoder_loss(decoder, img_embed_size, decoder.vocab, s)

    print(ans4)
    print(ans5)

    # Decoder shapes test
    grader.set_answer("rbpnH", ans4)
    # Decoder random loss test
    grader.set_answer("E2OIL", ans5)

    return


def part3(continue_epoch=None):
    s = reset_tf_session()
    # K.clear_session()

    train_captions_indexed, val_captions_indexed, vocab = get_captions_indexed()
    train_img_embeds, val_img_embeds = get_img_embeds()

    # init decoder
    print('img embed size', IMG_EMBED_SIZE)  # 2048
    decoder = DecoderV1(IMG_EMBED_SIZE)

    # init all variables
    s.run(tf.compat.v1.global_variables_initializer())

    # define optimizer operation to minimize the loss
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_step = optimizer.minimize(decoder.loss)

    batch_size = 64
    n_epochs = 12
    n_batches_per_epoch = 1000
    n_validation_batches = 100  # how many batches are used for validation after each epoch

    # actual training loop
    # to make training reproducible
    np.random.seed(42)
    random.seed(42)

    # will be used to save/load network weights.
    # you need to reset your default graph and define it in the same way to be able to load the saved weights!
    saver = tf.compat.v1.train.Saver()

    train_captions_indexed = np.array(train_captions_indexed)
    val_captions_indexed = np.array(val_captions_indexed)

    if continue_epoch is not None:
        epoch = continue_epoch
        saver.restore(s, get_checkpoint_path(epoch))
    else:
        epoch = 0

    while epoch < n_epochs:
        epoch += 1
        train_loss = 0
        counter = 0
        for _ in range(n_batches_per_epoch):
            train_loss += s.run([decoder.loss, train_step],
                                generate_batch(decoder,
                                               train_img_embeds,
                                               train_captions_indexed,
                                               batch_size,
                                               MAX_LEN))[0]
            counter += 1
            print("Training loss: %f" % (train_loss / counter))

        train_loss /= n_batches_per_epoch

        val_loss = 0
        for _ in range(n_validation_batches):
            val_loss += s.run(decoder.loss, generate_batch(decoder,
                                                           val_img_embeds,
                                                           val_captions_indexed,
                                                           batch_size,
                                                           MAX_LEN))
        val_loss /= n_validation_batches

        print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

        # save weights after finishing epoch
        save_path = get_checkpoint_path(epoch)
        print(f'Saving {epoch} weights to {save_path}')
        saver.save(s, save_path)

    print("Finished training!")

    ans6 = grading_utils.test_validation_loss(decoder, s, generate_batch, val_img_embeds, val_captions_indexed)

    print(ans6)

    # Validation loss
    grader.set_answer("YJR7z", ans6)

    return


def fix_path(filename):
    return os.path.expanduser(f'~/.keras/rl1/week6/{filename}')


def get_checkpoint_path(epoch=None):
    if epoch is None:
        return os.path.abspath(CHECKPOINT_ROOT + "weights")
    else:
        return os.path.abspath(CHECKPOINT_ROOT + "weights_{}".format(epoch))


def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model


def get_captions_indexed():
    global PAD_IDX

    # load prepared embeddings
    train_img_fns = utils.read_pickle(fix_path('train_img_fns.pickle'))
    val_img_fns = utils.read_pickle(fix_path('val_img_fns.pickle'))

    # check prepared samples of images
    list(filter(lambda x: x.endswith('_sample.zip'), os.listdir(fix_path('.'))))

    # extract captions from zip
    train_captions = get_captions_for_fns(train_img_fns, fix_path('captions_train-val2014.zip'),
                                          'annotations/captions_train2014.json')

    val_captions = get_captions_for_fns(val_img_fns, fix_path('captions_train-val2014.zip'),
                                        'annotations/captions_val2014.json')

    # # look at training example (each has 5 captions)
    # show_training_example(train_img_fns, train_captions, example_idx=142)

    # # preview captions data
    # print(train_captions[:2])

    # prepare vocabulary
    vocab = generate_vocabulary(train_captions)
    PAD_IDX = vocab[PAD]
    # vocab_inverse = {idx: w for w, idx in vocab.items()}
    # print('Vocab len', len(vocab))

    # make sure you use correct argument in caption_tokens_to_indices
    assert len(caption_tokens_to_indices(train_captions[:10], vocab)) == 10
    assert len(caption_tokens_to_indices(train_captions[:5], vocab)) == 5

    # replace tokens with indices
    train_captions_indexed = caption_tokens_to_indices(train_captions, vocab)
    val_captions_indexed = caption_tokens_to_indices(val_captions, vocab)

    return train_captions_indexed, val_captions_indexed, vocab


def get_img_embeds():
    # load prepared embeddings
    train_img_embeds = utils.read_pickle(fix_path('train_img_embeds.pickle'))
    val_img_embeds = utils.read_pickle(fix_path('val_img_embeds.pickle'))

    update_img_embed_size(train_img_embeds.shape[1])

    # # check shapes
    # print('Train img', train_img_embeds.shape, len(train_img_fns), len(train_captions))
    # print('Test img', val_img_embeds.shape, len(val_img_fns), len(val_captions))

    return train_img_embeds, val_img_embeds


def update_img_embed_size(new_size=None):
    global IMG_EMBED_SIZE

    if new_size is not None:
        IMG_EMBED_SIZE = new_size
    elif IMG_EMBED_SIZE is None:
        train_img_embeds = utils.read_pickle(fix_path('train_img_embeds.pickle'))
        IMG_EMBED_SIZE = train_img_embeds.shape[1]

    return IMG_EMBED_SIZE


def prepare_samples():
    # load pre-trained model
    encoder, preprocess_for_model = get_cnn_encoder()

    # extract train features
    train_img_embeds, train_img_fns = utils.apply_model(
            fix_path('train2014.zip'), encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    utils.save_pickle(train_img_embeds, "train_img_embeds.pickle")
    utils.save_pickle(train_img_fns, "train_img_fns.pickle")

    # extract validation features
    val_img_embeds, val_img_fns = utils.apply_model(
            fix_path('val2014.zip'), encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    utils.save_pickle(val_img_embeds, "val_img_embeds.pickle")
    utils.save_pickle(val_img_fns, "val_img_fns.pickle")

    sample_zip(fix_path('train2014.zip'), 'train2014_sample.zip')
    sample_zip(fix_path('val2014.zip'), 'val2014_sample.zip')


def sample_zip(fn_in, fn_out, rate=0.01, seed=42):
    np.random.seed(seed)
    with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
        sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
        for zInfo in sampled:
            fout.writestr(zInfo, fin.read(zInfo))


# extract captions from zip
def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]}
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))


# look at training example (each has 5 captions)
def show_training_example(train_img_fns, train_captions, example_idx=0):
    """
    You can change example_idx and see different images
    """
    zf = zipfile.ZipFile(fix_path('train2014_sample.zip'))
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    img = utils.decode_image_from_buf(zf.read(example))
    plt.imshow(utils.image_center_crop(img))
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()


# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))


def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more,
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary),
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    vocab = {PAD: 99, UNK: 99, START: 99, END: 99}
    for group in train_captions:
        for sentence in group:
            words = split_sentence(sentence)
            for word in words:
                vocab[word] = vocab.get(word, 0) + 1

    # Filter vocab to ones used 5+ times.
    vocab = dict(filter(lambda x: x[1] >= 5, vocab.items()))

    return {token: index for index, token in enumerate(sorted(vocab))}


def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Use `split_sentence` function to split sentence into tokens.
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    For the example above you should produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """

    res = []
    for group in captions:
        res_group = []
        for sentence in group:
            words = split_sentence(sentence)
            res_sentence = [vocab[START]]
            for word in words:
                res_sentence.append(vocab.get(word, vocab[UNK]))
            res_sentence.append(vocab[END])
            res_group.append(res_sentence)
        res.append(res_group)

    return res


# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    Try to use numpy, we need this function to be fast!
    """
    def create_rectangle_matrix(data):
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.full(mask.shape, fill_value=pad_idx)
        out[mask] = np.concatenate(data)
        return out

    # First convert data to square matrix
    matrix = create_rectangle_matrix(batch_captions)

    # Next fix padding
    len_target = max(map(len, batch_captions))
    if max_len is not None:
        len_target = min(max_len, len_target)
    len_diff = len_target - len(matrix[0])
    if len_diff > 0:
        matrix = np.pad(matrix, ((0, 0), (0, len_diff)), 'constant', constant_values=pad_idx)
    elif len_diff < 0:
        matrix = matrix[0:, 0:len_diff]

    return matrix


# generate batch via random sampling of images and captions for them,
# we use `max_len` parameter to control the length of the captions (truncating long captions)
def generate_batch(decoder, images_embeddings, indexed_captions, batch_size, max_len=None):
    """
    `images_embeddings` is a np.array of shape [number of images, IMG_EMBED_SIZE].
    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.
    Take random images and choose one random caption for each image.
    Remember to use `batch_captions_to_matrix` for padding and respect `max_len` parameter.
    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.
    """
    print('Image embeddings shape', images_embeddings.shape)
    print('Image captions length (group/sentence/word)', len(indexed_captions), len(indexed_captions[0]), len(indexed_captions[0][0]))
    print('Batch size', batch_size)

    idxs = np.random.choice(a=range(len(indexed_captions)), size=batch_size)
    batch_image_embeddings = np.take(images_embeddings, idxs, axis=0)  ### YOUR CODE HERE ###
    batch_captions = []
    for idx in idxs:
        batch_captions.append(np.random.choice(indexed_captions[idx]))
    batch_captions_matrix = batch_captions_to_matrix(batch_captions, PAD_IDX, max_len)  ### YOUR CODE HERE ###

    return {decoder.img_embeds: batch_image_embeddings,
            decoder.sentences: batch_captions_matrix}


class DecoderV1:
    def __init__(self, img_embed_size):
        self.img_embeds = None
        self.sentences = None
        self.img_embed_size = img_embed_size
        self.h0 = None
        self.word_embeds = None
        self.flat_hidden_states = None
        self.flat_token_logits = None
        self.flat_ground_truth = None
        self.flat_loss_mask = None
        self.loss = None
        self.vocab = None
        self.prepare()

    def prepare(self):
        train_captions_indexed, val_captions_indexed, vocab = get_captions_indexed()

        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        img_embeds = tf.compat.v1.placeholder(dtype='float32', shape=[None, self.img_embed_size])
        # img_embeds = keras.Input(dtype='float32', shape=(None, IMG_EMBED_SIZE))

        # [batch_size, time steps] of word ids
        sentences = tf.compat.v1.placeholder(dtype='int32', shape=[None, None])
        # sentences = keras.Input(dtype='int32', shape=(None, None))

        pad_idx_tf = tf.constant(vocab[PAD])
        end_idx_tf = tf.constant(vocab[END])

        # we use bottleneck here to reduce the number of parameters
        # image embedding -> bottleneck
        img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK,
                                          input_shape=(None, self.img_embed_size),
                                          activation='elu')
        # image embedding bottleneck -> lstm initial state
        img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                             input_shape=(None, IMG_EMBED_BOTTLENECK),
                                             activation='elu')

        # initial lstm cell state of shape (None, LSTM_UNITS),
        # we need to condition it on `img_embeds` placeholder.
        c0 = h0 = img_embed_bottleneck_to_h0(
                img_embed_to_bottleneck(img_embeds)
        )  ### YOUR CODE HERE ###  2x Dense on placeholder

        # word -> embedding
        word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
        # lstm cell (from tensorflow)
        lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(LSTM_UNITS)

        # embed all tokens but the last for lstm input,
        # remember that L.Embedding is callable,
        # use `sentences` placeholder as input.
        word_embeds = word_embed(sentences[:, :-1])  ### YOUR CODE HERE ###  Dense on placeholder

        # during training we use ground truth tokens `word_embeds` as context for next token prediction.
        # that means that we know all the inputs for our lstm and can get
        # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
        # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
        hidden_states, _ = tf.compat.v1.nn.dynamic_rnn(lstm, word_embeds,
                                                       initial_state=tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c0, h0))

        # now we need to calculate token logits for all the hidden states

        # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
        flat_hidden_states = tf.reshape(hidden_states, (-1, LSTM_UNITS))  ### YOUR CODE HERE ###

        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK,
                                          input_shape=(None, LSTM_UNITS),
                                          activation="elu")
        # logits bottleneck -> logits for next token prediction
        token_logits = L.Dense(len(vocab),
                               input_shape=(None, LOGIT_BOTTLENECK))

        # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
        flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))  ### YOUR CODE HERE ###  2x Dense on reshaped LSTM output

        # then, we flatten the ground truth token ids.
        # remember, that we predict next tokens for each time step,
        # use `sentences` placeholder.
        flat_ground_truth = tf.reshape(sentences[:, 1:], (-1, ))  ### YOUR CODE HERE ###

        # we need to know where we have real tokens (not padding) in `flat_ground_truth`,
        # we don't want to propagate the loss for padded output tokens,
        # fill `flat_loss_mask` with 1.0 for real tokens (not pad_idx) and 0.0 otherwise.
        flat_loss_mask = tf.math.not_equal(flat_ground_truth, pad_idx_tf)  ### YOUR CODE HERE ###

        # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=flat_ground_truth,
                logits=flat_token_logits
        )

        # compute average `xent` over tokens with nonzero `flat_loss_mask`.
        # we don't want to account misclassification of PAD tokens, because that doesn't make sense,
        # we have PAD tokens for batching purposes only!
        loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask))  ### YOUR CODE HERE ###

        self.vocab = vocab                              # Vocab keys w/ freq >= 5 (& padding)

        self.img_embeds = img_embeds                    # Placeholder
        self.h0 = h0                                    # 2x Dense on img_embeds placeholder

        self.sentences = sentences                      # Placeholder
        self.word_embeds = word_embeds                  # 1x Dense on sentences placeholder
        self.flat_hidden_states = flat_hidden_states    # LSTM on word_embeds w/ LSTMCell & init st h0/c0, Reshaped to LSTM_UNITS
        self.flat_token_logits = flat_token_logits      # 2x Dense on FHS
        self.flat_ground_truth = flat_ground_truth      # Reshaped of sentences placeholder
        self.flat_loss_mask = flat_loss_mask            # FGT w/ padding masked out
        self.loss = loss                                # Softmax CE logits loss


if __name__ == "__main__":
    # execute only if run as a script
    main()
