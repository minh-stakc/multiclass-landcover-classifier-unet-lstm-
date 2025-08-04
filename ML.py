import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (ConvLSTM2D, BatchNormalization, Conv2D, concatenate,
                                     Input, Lambda, Activation, GlobalAveragePooling3D, Reshape, LayerNormalization, 
                                     Dense, MultiHeadAttention, Input, Conv3D, GlobalAveragePooling3D, Dense, TimeDistributed, 
                                     MaxPooling2D, Conv2DTranspose, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, base_path, tiles, chips, batch_size=1, img_size=(256, 256), shuffle=True):
        self.base_path = base_path  # 'landcovernet_as/data/v1.0/2018'
        self.tiles = tiles            # ['36SWF', ...]
        self.chips = chips          # ['00', '01', ...]
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.chips))
        self.BANDS_S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12"]
        self.BANDS_L8 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07"]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.chips) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch_S2, X_batch_S1, X_batch_L8, y_batch = [], [], [], []

        for idx in batch_indexes:
            chip = self.chips[idx]
            x, y = None, None

            for tile in self.tiles:
                x, y = self.load_chip(tile, chip)

            if x is None or y is None:
                print(f"Skipping index {idx}")
                return self.__getitem__((index + 1) % self.__len__())

            # x = [x[i].astype(np.float32) / 255.0 for i in [0,1,2]]
            x[0] = np.clip(x[0] / 20000, 0.0, 1.0)
            x[1] = np.clip(x[1] / 15, 0.0, 1.0)
            x[2] = np.clip(x[2] / 1.6, 0.0, 1.0)


            # One-hot encode label
            if y.ndim == 2:
                y[y > 7] = 0
                y = tf.keras.utils.to_categorical(y, num_classes=8)
            else:
                raise ValueError(f"Label at index {idx} has unexpected shape: {y.shape}")

            X_batch_S2.append(x[0])
            X_batch_S1.append(x[1])
            X_batch_L8.append(x[2])
            y_batch.append(y)

        if len(X_batch_S2) == 0 or len(X_batch_S1) == 0 or len(X_batch_L8) == 0:
            raise ValueError(f"All samples in batch {index} were invalid")

        return [np.array(X_batch_S2), np.array(X_batch_S1), np.array(X_batch_L8)], np.array(y_batch)


    def load_chip(self, tile, chip):
        chip_dir = os.path.join(self.base_path, tile, chip) #D:/LandCoverNet_2/landcovernet_as/data/v1.0/2018/36SWF/00
        label_path = os.path.join(chip_dir, f"{tile}_{chip}_2018_LC_10m.tif")
        csv_path = os.path.join(chip_dir, f"{tile}_{chip}_labeling_dates.csv")

        if not os.path.exists(csv_path) or not os.path.exists(label_path):
            return None, None

        try:
            dates = pd.read_csv(csv_path, header=None)[1].astype(str).tolist()
            dates = dates[1:]
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return None, None

        samples = []
        for date in dates:
            try:
                bands = [self.read_tif(self.get_band_path(tile, chip, date, b)) for b in self.BANDS_S2]
                band_stack = np.stack(bands, axis=-1)
                samples.append(band_stack)
            except Exception as e:
                print(f"Failed chip {chip} on {date}: {e}")
                continue

        if len(samples) < 24:
            print(f"Not enough valid scenes for {chip}")
            return None, None

        S1_stack = []
        L8_stack = []
        S1_path = os.path.join(chip_dir, "S1")
        L8_path = os.path.join(chip_dir, "L8")
        for i in np.linspace(0, len(os.listdir(S1_path)) - 1, 12, dtype=int):
            VH = self.read_tif(os.path.join(S1_path, f"{os.listdir(S1_path)[i]}", f"{os.listdir(S1_path)[i]}_VH_10m.tif"))
            VV = self.read_tif(os.path.join(S1_path, f"{os.listdir(S1_path)[i]}", f"{os.listdir(S1_path)[i]}_VV_10m.tif"))
            stack_temp = np.stack([VH, VV], axis=-1)
            S1_stack.append(stack_temp)
        for i in np.linspace(0, len(os.listdir(L8_path)) - 1, 12, dtype=int):
            try:
                folder = os.listdir(L8_path)[i]
                bands_L8 = []
                for b in self.BANDS_L8:
                    path = os.path.join(L8_path, folder, f"{folder}_{b}_10m.tif")
                    band = self.read_tif(path)
                    if np.isnan(band).any():
                        raise ValueError(f"NaNs in L8 band {b} of {path}")
                    bands_L8.append(band)
                stack_temp = np.stack(bands_L8, axis=-1)
                L8_stack.append(stack_temp)
            except Exception as e:
                print(f"[L8] Skipping {chip} scene {folder}: {e}")
                continue
        
        X = [np.stack(samples, axis=0), np.stack(S1_stack, axis=0), np.stack(L8_stack, axis=0)]
        label = self.read_tif(label_path)

        return X, label

    def read_tif(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            # Replace NaNs or nodata with 0 or some valid value
            arr[np.isnan(arr)] = 0.0
            if src.nodata is not None:
                arr[arr == src.nodata] = 0.0
            return arr


    def get_band_path(self, tile, chip, date, band):
        return os.path.join(
            self.base_path, tile, chip, "S2", f"{tile}_{chip}_{date}",
            f"{tile}_{chip}_{date}_{band}_10m.tif"
        )

chips = [f"{i:02d}" for i in range(30)]
tiles = ["36SWF", "36SXH", "37SFU", "37TFJ"]
np.random.seed(42)
np.random.shuffle(chips)

split_idx_1 = int(0.7 * len(chips))
split_idx_2 = int(0.9 * len(chips))
train_chips = chips[:split_idx_1]
val_chips = chips[split_idx_1:split_idx_2]
test_chips = chips[split_idx_2:]
BASE_PATH = "D:/LandCoverNet_2/landcovernet_as/data/v1.0/2018"

train_generator = DataGenerator(base_path=BASE_PATH, tiles=tiles, chips=train_chips, batch_size=1, shuffle=True)
val_generator = DataGenerator(base_path=BASE_PATH, tiles=tiles, chips=val_chips, batch_size=1, shuffle=False)
test_generator = DataGenerator(base_path=BASE_PATH, tiles=tiles, chips=test_chips, batch_size=1, shuffle=False)

x, y = train_generator[0]
print("Train:")
print("X shape:", x[0].shape, x[1].shape, x[2].shape)
print("Y shape:", y.shape)
print("NaNs:", np.isnan(x[0]).any(), np.isnan(x[1]).any(), np.isnan(x[2]).any(), np.isnan(y).any())
print("Max values:", np.max(x[0]), np.max(x[1]), np.max(x[2]), np.max(y))
print("Min values:", np.min(x[0]), np.min(x[1]), np.min(x[2]), np.min(y))

x, y = val_generator[0]
print("\nValidation:")
print("X shape:", x[0].shape, x[1].shape, x[2].shape)
print("Y shape:", y.shape)
print("NaNs:", np.isnan(x[0]).any(), np.isnan(x[1]).any(), np.isnan(x[2]).any(), np.isnan(y).any())
print("Max values:", np.max(x[0]), np.max(x[1]), np.max(x[2]), np.max(y))
print("Min values:", np.min(x[0]), np.min(x[1]), np.min(x[2]), np.min(y))

x, y = test_generator[0]
print("\nTest:")
print("X shape:", x[0].shape, x[1].shape, x[2].shape)
print("Y shape:", y.shape)
print("NaNs:", np.isnan(x[0]).any(), np.isnan(x[1]).any(), np.isnan(x[2]).any(), np.isnan(y).any())
print("Max values:", np.max(x[0]), np.max(x[1]), np.max(x[2]), np.max(y))
print("Min values:", np.min(x[0]), np.min(x[1]), np.min(x[2]), np.min(y))

    
    
def unet_conv_lstm(initial_filter=16):
    s2_input = Input(shape=(24, 256, 256, 11), name='s2_input')
    s1_input = Input(shape=(12, 256, 256, 2), name='s1_input')
    l8_input = Input(shape=(12, 256, 256, 7), name='l8_input')

    # Initial 3D conv features
    s1_feat = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(s1_input)
    l8_feat = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(l8_input)
    
    # Global context features from S1 and L8
    s1_summary = GlobalAveragePooling3D()(s1_feat)
    l8_summary = GlobalAveragePooling3D()(l8_feat)
    aux_features = tf.keras.layers.Concatenate()([s1_summary, l8_summary])

    # Encoder
    c1 = ConvLSTM2D(initial_filter, (3, 3), padding='same', return_sequences=True)(s2_input)
    c1 = LayerNormalization()(c1)
    c1_pool = TimeDistributed(MaxPooling2D((2, 2)))(c1)

    c2 = ConvLSTM2D(initial_filter*2, (3, 3), padding='same', return_sequences=True)(c1_pool)
    c2 = LayerNormalization()(c2)
    c2_pool = TimeDistributed(MaxPooling2D((2, 2)))(c2)

    c3 = ConvLSTM2D(initial_filter*4, (3, 3), padding='same', return_sequences=True)(c2_pool)
    c3 = LayerNormalization()(c3)
    c3_pool = TimeDistributed(MaxPooling2D((2, 2)))(c3)

    c4 = ConvLSTM2D(initial_filter*8, (3, 3), padding='same', return_sequences=True)(c3_pool)
    c4 = LayerNormalization()(c4)
    c4_pool = TimeDistributed(MaxPooling2D((2, 2)))(c4)

    # Bottleneck
    b = ConvLSTM2D(initial_filter*16, (3, 3), padding='same', return_sequences=True)(c4_pool)
    b = LayerNormalization()(b)
    
    b_shape = tf.shape(b)
    b_flat = tf.reshape(b, (b_shape[0], -1, b_shape[-1]))
    b_flat = LayerNormalization()(b_flat)
    
    aux_dense = Dense(b_flat.shape[-1])(aux_features)
    aux_expanded = tf.expand_dims(aux_dense, axis=1)
    aux_tiled = tf.tile(aux_expanded, [1, tf.shape(b_flat)[1], 1])
    print(aux_tiled.shape)
    
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(
        query=b_flat, key=aux_tiled, value=aux_tiled
    )
    attn = tf.reshape(attn, tf.shape(b))
    b = tf.concat([b, attn], axis=-1)
    print(b.shape)

    # Decoder
    u4 = TimeDistributed(Conv2DTranspose(initial_filter*8, (2, 2), strides=(2, 2), padding='same'))(b)
    u4 = concatenate([u4, c4])
    u4 = ConvLSTM2D(initial_filter*8, (3, 3), padding='same', return_sequences=True)(u4)

    u3 = TimeDistributed(Conv2DTranspose(initial_filter*4, (2, 2), strides=(2, 2), padding='same'))(u4)
    u3 = concatenate([u3, c3])
    u3 = ConvLSTM2D(initial_filter*4, (3, 3), padding='same', return_sequences=True)(u3)

    u2 = TimeDistributed(Conv2DTranspose(initial_filter*2, (2, 2), strides=(2, 2), padding='same'))(u3)
    u2 = concatenate([u2, c2])
    u2 = ConvLSTM2D(initial_filter*2, (3, 3), padding='same', return_sequences=True)(u2)

    u1 = TimeDistributed(Conv2DTranspose(initial_filter, (2, 2), strides=(2, 2), padding='same'))(u2)
    u1 = concatenate([u1, c1])
    u1 = ConvLSTM2D(initial_filter, (3, 3), padding='same', return_sequences=True)(u1)

    # Output
    o1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same')(u1)
    o1 = Lambda(lambda t: t[:, -1, ...])(o1)
    output = Activation("softmax")(o1)

    model = Model(inputs=[s2_input, s1_input, l8_input], outputs=output)
    return model


def masked_categorical_crossentropy(ignore_class=0):
    def loss(y_true, y_pred):
        mask = tf.not_equal(tf.argmax(y_true, axis=-1), ignore_class)
        mask = tf.cast(mask, tf.float32)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def per_class_accuracy(num_classes=8, ignore_class=0):
    def metric(y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        mask = tf.not_equal(y_true, ignore_class)

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        per_class_acc = []
        for i in range(num_classes):
            if i == ignore_class:
                continue
            match = tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i))
            total = tf.equal(y_true, i)
            acc = tf.reduce_sum(tf.cast(match, tf.float32)) / (tf.reduce_sum(tf.cast(total, tf.float32)) + 1e-6)
            per_class_acc.append(acc)

        return tf.reduce_mean(per_class_acc)
    return metric

def train_model(model, train_generator, val_data, epochs=50):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=masked_categorical_crossentropy(ignore_class=0), 
        metrics=['accuracy', per_class_accuracy(num_classes=8, ignore_class=0), tf.keras.metrics.MeanIoU(num_classes=8)]
    )
    model.summary()
    model.fit(train_generator, validation_data=val_data, epochs=epochs,)
    return model



model = unet_conv_lstm()
trained_model = train_model(model, train_generator, val_generator)

val_loss, val_accuracy = model.evaluate(test_generator)
print(f"Validation loss: {val_loss}, accuracy: {val_accuracy}")