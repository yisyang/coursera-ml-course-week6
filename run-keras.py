import os
from tensorflow import keras
import numpy as np
import utils
import click
from decoder import DecoderV2, EncoderV2
from matplotlib import pyplot as plt

L = keras.layers
K = keras.backend
IMG_SIZE = 299

# special tokens
PAD = "#000#"           # Hack, to make padding token at index 0 after sorting.
UNK = "#UNK#"
START = "#START#"
END = "#END#"
PAD_IDX = 0

BATCH_SIZE = 64
IMG_EMBED_SIZE = 0      # Placeholder
SENTENCE_MAX_LEN = 20   # Truncate long captions to speed up training


@click.command()
@click.option('-m', '--mode', default='train', type=click.Choice(['train', 'validate', 'predict', 't', 'v', 'p']))
@click.option('-l', '--load-checkpoint', type=click.types.INT, default=0, help='Integer of episode checkpoint to load.')
def main(mode: str = 'train', load_checkpoint: int = 0):
    if mode in ['train', 't']:
        train(load_checkpoint)
    elif mode in ['validate', 'v']:
        validate(load_checkpoint)
    elif mode in ['predict', 'p']:
        predict(load_checkpoint)
    else:
        test(load_checkpoint)


def train(continue_epoch: int = None):
    K.clear_session()

    train_captions_indexed, val_captions_indexed, vocab = utils.get_captions_indexed()
    train_img_embeds, val_img_embeds = utils.get_img_embeds()
    update_img_embed_size(train_img_embeds.shape[1])

    n_epochs = 24
    n_batches_per_epoch = 500
    n_validation_batches = 10  # how many batches are used for validation after each epoch

    epoch = continue_epoch or 0

    # init decoder
    # print('img embed size', IMG_EMBED_SIZE)  # 2048
    decoder = DecoderV2(vocab, IMG_EMBED_SIZE, epoch)
    model = decoder.model
    print(model.summary())

    # actual training loop
    # to make training reproducible
    np.random.seed(42)

    train_captions_indexed = np.array(train_captions_indexed)
    val_captions_indexed = np.array(val_captions_indexed)

    while epoch < n_epochs:
        epoch += 1
        train_loss = 0
        for batch_num in range(n_batches_per_epoch):
            img_embeds, sentences = generate_batch(train_img_embeds,
                                                   train_captions_indexed,
                                                   BATCH_SIZE,
                                                   SENTENCE_MAX_LEN)
            # print(img_embeds.shape)                               # (batch_size, 2048), batch_size=64
            # print('Sentences shape', sentences.shape)             # (batch_size, 20)
            # print('Sentence -1 shape', sentences[:, 1:].shape)    # (batch_size, 19)
            history = model.fit(x=[img_embeds, sentences[:, :-1]], y=sentences[:, 1:], verbose=0)
            # print(history.history)
            train_loss += history.history['loss'][0]
            print(f'Epoch {epoch}  Batch {batch_num}/{n_batches_per_epoch}  Loss {history.history["loss"][0]}',
                  end='\r')
        train_loss /= n_batches_per_epoch

        print(f'Epoch: {epoch}  Train loss: {train_loss:.4}')

        validate(epoch, n_validation_batches, model, val_img_embeds, val_captions_indexed, vocab)

        # save weights after finishing epoch
        decoder.save_weights(epoch)

    print("Finished training!")

    return


def validate(continue_epoch, n_batches=50, model=None, val_img_embeds=None, val_captions_indexed=None, vocab=None):
    epoch = continue_epoch or 0

    if val_captions_indexed is None or vocab is None:
        _, val_captions_indexed, vocab = utils.get_captions_indexed(True)
    if val_img_embeds is None:
        _, val_img_embeds = utils.get_img_embeds(True)
    if model is None:
        update_img_embed_size(val_img_embeds.shape[1])

        decoder = DecoderV2(vocab, IMG_EMBED_SIZE, epoch)
        model = decoder.model

    val_loss = 0
    for _ in range(n_batches):
        img_embeds, sentences = generate_batch(val_img_embeds,
                                               val_captions_indexed,
                                               BATCH_SIZE,
                                               SENTENCE_MAX_LEN)
        val_loss += model.evaluate(x=[img_embeds, sentences[:, :-1]], y=sentences[:, 1:], verbose=0)
    val_loss /= n_batches

    print(f'Epoch: {epoch}  Val loss: {val_loss:.4}')


def test(continue_epoch):
    epoch = continue_epoch or 0
    _, val_captions_indexed, vocab = utils.get_captions_indexed(True)
    _, val_img_embeds = utils.get_img_embeds(True)
    update_img_embed_size(val_img_embeds.shape[1])
    decoder = DecoderV2(vocab, IMG_EMBED_SIZE, epoch)
    model = decoder.model
    batch_size = 1
    img_embeds, sentences = generate_batch(val_img_embeds,
                                           val_captions_indexed,
                                           batch_size,
                                           SENTENCE_MAX_LEN)

    start_idx = vocab[START]
    end_idx = vocab[END]

    vocab_inverse = {idx: w for w, idx in vocab.items()}

    # Start with #START# and keep predicting until #END#
    words_in = np.full(shape=(batch_size, 1), fill_value=start_idx)         # shape 1 1
    pred = [[start_idx]]                                                    # placeholder
    while pred[0][-1] != end_idx:
        y = model.predict(x=[img_embeds, words_in])                         # shape 1 i 8769
        pred = np.argmax(y, axis=2)                                         # shape 1 i
        words_in = np.insert(pred, 0, start_idx, axis=1)                    # shape 1 i+1
        print(pred[0], end='\r')

    print('')

    words_pred = [vocab_inverse[v] for v in pred[0] if v > 3]
    print('pred: ', ' '.join(words_pred))

    words_val = [vocab_inverse[v] for v in sentences[0] if v > 3]
    print('val: ', ' '.join(words_val))


def predict(continue_epoch):
    epoch = continue_epoch or 0
    encoder = EncoderV2()

    # look at validation prediction example
    def apply_model_to_image_raw_bytes(raw):
        img = utils.decode_image_from_buf(raw)
        # plt.figure(figsize=(7, 7))
        # plt.grid('off')
        # plt.axis('off')
        # plt.imshow(img)
        # plt.show()
        img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), encoder.preprocessor)
        return img

    _, val_captions_indexed, vocab = utils.get_captions_indexed(True)
    update_img_embed_size(2048)                                     # Optimize
    decoder = DecoderV2(vocab, IMG_EMBED_SIZE, epoch)
    model = decoder.model
    batch_size = 1
    start_idx = vocab[START]
    end_idx = vocab[END]

    vocab_inverse = {idx: w for w, idx in vocab.items()}

    examples_dir = './examples'
    for filename in os.listdir(examples_dir):
        filepath = os.path.join(examples_dir, filename)
        print(f'Now processing {filepath}.')
        with open(filepath, 'rb') as fh:
            img_i = apply_model_to_image_raw_bytes(fh.read())
            img_i = img_i.reshape(1, IMG_SIZE, IMG_SIZE, 3)
            # print(img_i.shape)                                    # 1 299 299 3
            img_embeds = encoder.model.predict(img_i)
            # print(img_embeds.shape)                               # 1 2048

        # Start with #START# and keep predicting until #END#
        words_in = np.full(shape=(batch_size, 1), fill_value=start_idx)         # shape 1 1
        pred = [[start_idx]]                                                    # placeholder
        while pred[0][-1] != end_idx and len(pred[0]) < 25:
            y = model.predict(x=[img_embeds, words_in])                         # shape 1 i 8769
            pred = np.argmax(y, axis=2)                                         # shape 1 i
            words_in = np.insert(pred, 0, start_idx, axis=1)                    # shape 1 i+1
            print(pred[0], end='\r')

        print('')

        words_pred = [vocab_inverse[v] for v in pred[0] if v > 3]
        print('pred: ', ' '.join(words_pred))


def update_img_embed_size(new_size=None):
    global IMG_EMBED_SIZE

    if new_size is not None:
        IMG_EMBED_SIZE = new_size
    elif IMG_EMBED_SIZE == 0:
        train_img_embeds = utils.read_pickle('train_img_embeds.pickle')
        IMG_EMBED_SIZE = train_img_embeds.shape[1]

    return IMG_EMBED_SIZE


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
        mask = np.arange(lens.max(initial=-1)) < lens[:, None]

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
def generate_batch(images_embeddings, indexed_captions, batch_size, max_len=None):
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
    # print('Image embeddings shape', images_embeddings.shape)
    # print('Image captions length (group/sentence/word)', len(indexed_captions), len(indexed_captions[0]),
    #       len(indexed_captions[0][0]))
    # print('Batch size', batch_size)
    # Image embeddings shape (82783, 2048)
    # Image captions length (group/sentence/word) 82783 5 10
    # Batch size 64

    indices = np.random.choice(a=range(len(indexed_captions)), size=batch_size, replace=False)
    batch_image_embeddings = np.take(images_embeddings, indices, axis=0)
    batch_captions = []
    for image_index in indices:
        caption_index = np.random.randint(0, len(indexed_captions[image_index]))
        batch_captions.append(indexed_captions[image_index][caption_index])
    batch_captions_matrix = batch_captions_to_matrix(batch_captions, PAD_IDX, max_len)

    return batch_image_embeddings, batch_captions_matrix


if __name__ == "__main__":
    # execute only if run as a script
    main()
