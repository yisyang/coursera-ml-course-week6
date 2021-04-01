import os
from tensorflow import keras
import utils


L = keras.layers
K = keras.backend

IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
CHECKPOINT_ROOT = "checkpoints/"


class DecoderV2:
    def __init__(self, vocab, img_embed_size: int, continue_epoch: int = 0, batch_size=None):
        self.vocab = vocab
        self.img_embed_size = img_embed_size
        self.continue_epoch = continue_epoch
        self.epoch = self.continue_epoch
        self.batch_size = batch_size
        self.prepared = False
        self.model = None
        self.prepare()

    @staticmethod
    def get_checkpoint_path(epoch: int):
        return os.path.abspath(CHECKPOINT_ROOT + f'weights_{epoch:04}')

    def load_weights(self, epoch: int):
        save_path = self.get_checkpoint_path(epoch)
        print(f'Loading {epoch} weights from {save_path}')
        self.model.load_weights(save_path)

    def save_weights(self, epoch):
        save_path = self.get_checkpoint_path(epoch)
        print(f'Saving {epoch} weights to {save_path}')
        self.model.save_weights(save_path)

    def prepare(self):
        if self.prepared:
            return self.model

        train_captions_indexed, val_captions_indexed, vocab = utils.get_captions_indexed()

        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        img_embed_in = L.Input(dtype='float', shape=self.img_embed_size, name='image', batch_size=self.batch_size)
        # We use bottleneck here to reduce the number of parameters
        # image embedding -> bottleneck
        d1 = L.Dense(units=IMG_EMBED_BOTTLENECK, activation='elu')
        # image embedding bottleneck -> lstm initial state
        d2 = L.Dense(units=LSTM_UNITS, activation='elu')
        c0 = h0 = d2(d1(img_embed_in))
        # print('c0 shape', c0.shape)

        # word -> embedding
        # Embed all tokens but the last for lstm input,
        captions_in = L.Input(dtype='int32', shape=[None], name='caption', batch_size=self.batch_size)
        # During training we use ground truth tokens `word_embeds` as context for next token prediction.
        # that means that we know all the inputs for our lstm and can get
        # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
        # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
        words = L.Embedding(input_dim=len(vocab), output_dim=WORD_EMBED_SIZE, mask_zero=True)(captions_in)

        # Reduce words features into LSTM_UNITS vector
        words_features = L.LSTM(units=LSTM_UNITS, stateful=False, return_sequences=True)(words, initial_state=(c0, h0))

        # Now we need to calculate token logits for all the hidden states
        # We use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        lstm2bottleneck = L.Dense(LOGIT_BOTTLENECK, activation="elu")
        # Logits bottleneck -> logits for next token prediction
        bottleneck2logit = L.Dense(len(vocab), name='logits')
        # out = bottleneck2logit(lstm2bottleneck(reshape(words_features)))
        out = bottleneck2logit(lstm2bottleneck(words_features))     # 64, 8769

        model = keras.Model(inputs=[img_embed_in, captions_in], outputs=out)
        model.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adamax(lr=0.001)
        )
        self.model = model

        # Load weights as needed.
        if self.continue_epoch > 0:
            self.load_weights(self.continue_epoch)

        self.prepared = True

        return


class EncoderV2:
    def __init__(self):
        self.prepared = False
        self.model = None
        self.preprocessor = None
        self.prepare()
        return

    # we take the last hidden layer of IncetionV3 as an image embedding
    def prepare(self):
        K.set_learning_phase(False)
        app = keras.applications.InceptionV3(include_top=False)
        out = L.GlobalAveragePooling2D()(app.output)

        self.model = keras.Model(inputs=app.inputs, outputs=out)
        self.preprocessor = keras.applications.inception_v3.preprocess_input

        self.prepared = True

        return
