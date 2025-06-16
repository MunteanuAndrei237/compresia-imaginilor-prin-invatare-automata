import tensorflow as tf
from tensorflow_compression.python.layers.gdn import GDN

def autoencoder(image_size = 256):
    input = tf.keras.Input(shape=(image_size, image_size, 3))

    # Encoder
    conv2d = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(input)
    conv2d_1 = tf.keras.layers.Conv2D(32, (3, 3), 2, padding='same')(conv2d)
    gdn = GDN()(conv2d_1)
    conv2d_2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(gdn)
    conv2d_3 = tf.keras.layers.Conv2D(32, (3, 3), 2, padding='same')(conv2d_2)
    gdn_1 = GDN()(conv2d_3)  # Apply GDN
    conv2d_4 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(gdn_1)
    conv2d_5 = tf.keras.layers.Conv2D(32, (3, 3), 2, padding='same')(conv2d_4)
    gdn_2 = GDN()(conv2d_5)  # Apply GDN
    conv2d_6 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(gdn_2)

    # Flatten and reshape latent values
    flatten = tf.keras.layers.Flatten()(conv2d_6)
    reshape = tf.keras.layers.Reshape((32, 32, 32))(flatten)

    # Decoder
    conv2d_transpose = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same')(reshape)
    conv2d_transpose_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), 2, padding='same')(conv2d_transpose)
    igdn = GDN(inverse=True)(conv2d_transpose_1)
    conv2d_transpose_2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same')(igdn)
    conv2d_transpose_3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), 2, padding='same')(conv2d_transpose_2)
    igdn_1 = GDN(inverse=True)(conv2d_transpose_3)
    conv2d_transpose_4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same')(igdn_1)
    conv2d_transpose_5 = tf.keras.layers.Conv2DTranspose(32, (3, 3), 2, padding='same')(conv2d_transpose_4)
    igdn_2 = GDN(inverse=True)(conv2d_transpose_5) # Apply inverse GDN
    conv2d_transpose_6 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same')(igdn_2)

    # Back to 3 color channels
    output = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(conv2d_transpose_6)
    return tf.keras.Model(input, [flatten, output])