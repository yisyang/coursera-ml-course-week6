#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import queue
import threading
import cv2
import numpy as np
import pickle
import zipfile
import json
from collections import defaultdict
import re


# special tokens
PAD = "#000#"   # Hack, to make padding token at index 0 after sorting.
UNK = "#UNK#"
START = "#START#"
END = "#END#"


def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]


def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def crop_and_preprocess(img, input_shape, preprocess_for_model):
    img = image_center_crop(img)  # take center crop
    img = cv2.resize(img, input_shape)  # resize for our model
    img = img.astype("float32")  # prepare for normalization
    img = preprocess_for_model(img)  # preprocess for model
    return img


def apply_model(zip_fn, model, preprocess_for_model, extensions=(".jpg",), input_shape=(224, 224), batch_size=32):
    # queue for cropped images
    q = queue.Queue(maxsize=batch_size * 10)

    # when read thread put all images in queue
    read_thread_completed = threading.Event()

    # time for read thread to die
    kill_read_thread = threading.Event()

    def reading_thread(zip_fn):
        zf = zipfile.ZipFile(zip_fn)
        for fn in zf.namelist():
            if kill_read_thread.is_set():
                break
            if os.path.splitext(fn)[-1] in extensions:
                buf = zf.read(fn)  # read raw bytes from zip for fn
                img = decode_image_from_buf(buf)  # decode raw bytes
                img = crop_and_preprocess(img, input_shape, preprocess_for_model)
                while True:
                    try:
                        q.put((os.path.split(fn)[-1], img), timeout=1)  # put in queue
                    except queue.Full:
                        if kill_read_thread.is_set():
                            break
                        continue
                    break

        read_thread_completed.set()  # read all images

    # start reading thread
    t = threading.Thread(target=reading_thread, args=(zip_fn,))
    t.daemon = True
    t.start()

    img_fns = []
    img_embeddings = []

    batch_imgs = []

    def process_batch(imgs):
        imgs = np.stack(imgs, axis=0)
        batch_embeddings = model.predict(imgs)
        img_embeddings.append(batch_embeddings)

    try:
        while True:
            try:
                fn, img = q.get(timeout=1)
            except queue.Empty:
                if read_thread_completed.is_set():
                    break
                continue
            img_fns.append(fn)
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size:
                process_batch(batch_imgs)
                batch_imgs = []
            q.task_done()
        # process last batch
        if len(batch_imgs):
            process_batch(batch_imgs)
    finally:
        kill_read_thread.set()
        t.join()

    q.join()

    img_embeddings = np.vstack(img_embeddings)
    return img_embeddings, img_fns


def save_pickle(obj, fn):
    with open(fix_path(fn), "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fn):
    with open(fix_path(fn), "rb") as f:
        return pickle.load(f)


def fix_path(filename):
    return os.path.expanduser(f'~/.keras/rl1/week6/{filename}')


def get_captions_indexed(validate_only=False):
    # load prepared embeddings
    train_img_fns = read_pickle('train_img_fns.pickle')
    val_img_fns = read_pickle('val_img_fns.pickle')

    # check prepared samples of images
    # list(filter(lambda x: x.endswith('_sample.zip'), os.listdir(fix_path('.'))))

    # extract captions from zip
    train_captions = get_captions_for_fns(train_img_fns, fix_path('captions_train-val2014.zip'),
                                          'annotations/captions_train2014.json')

    val_captions = get_captions_for_fns(val_img_fns, fix_path('captions_train-val2014.zip'),
                                        'annotations/captions_val2014.json')

    # Prepare vocabulary
    vocab = generate_vocabulary(train_captions)
    # vocab_inverse = {idx: w for w, idx in vocab.items()}
    # print('Vocab len', len(vocab))

    # Make sure you use correct argument in caption_tokens_to_indices
    assert len(caption_tokens_to_indices(train_captions[:10], vocab)) == 10
    assert len(caption_tokens_to_indices(train_captions[:5], vocab)) == 5

    # Replace tokens with indices
    if validate_only:
        train_captions_indexed = None
    else:
        train_captions_indexed = caption_tokens_to_indices(train_captions, vocab)
    val_captions_indexed = caption_tokens_to_indices(val_captions, vocab)

    # # Check shapes
    # print('Train captions', len(train_img_fns), len(train_captions))      # 82783 82783
    # print('Test captions', len(val_img_fns), len(val_captions))
    # print('Vocab len', len(vocab))                                        # 8769

    return train_captions_indexed, val_captions_indexed, vocab


def get_img_embeds(validate_only=False):
    # Load prepared embeddings
    if validate_only:
        train_img_embeds = None
    else:
        train_img_embeds = read_pickle('train_img_embeds.pickle')
    val_img_embeds = read_pickle('val_img_embeds.pickle')

    # # Check shapes
    # print('Train img', train_img_embeds.shape)
    # print('Test img', val_img_embeds.shape)

    return train_img_embeds, val_img_embeds


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


# split sentence into tokens (split into lowercase words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\\W+', sentence.lower())))


def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more,
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary),
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    vocab_dict = {PAD: 99, UNK: 99, START: 99, END: 99}
    for group in train_captions:
        for sentence in group:
            words = split_sentence(sentence)
            for word in words:
                vocab_dict[word] = vocab_dict.get(word, 0) + 1

    # Filter vocab to ones used 5+ times.
    vocab_dict = {k: v for k, v in vocab_dict.items() if v >= 5}

    # Map keys to indices.
    vocab = {token: index for index, token in enumerate(sorted(vocab_dict))}

    return vocab


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
