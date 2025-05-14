import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys

sys.stdout = open("result.txt", "w")
dataset, info = tfds.load("tf_flowers", with_info=True, as_supervised=True)

validation_dataset = dataset['train'].take(320)
train_dataset = dataset['train'].skip(320).take(3200)

def preprocess_image(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    return image, image

train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(10).shuffle(1000)
validation_dataset = validation_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(10)

input_img = tf.keras.Input(shape=(256, 256, 3))

x = tf.keras.layers.Conv2D(128, (3, 3), activation='leaky_relu', strides=2, padding='same')(input_img)
x = tf.keras.layers.Conv2D(256, (3, 3), activation='leaky_relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2D(512, (3, 3), activation='leaky_relu', strides=2, padding='same')(x)

x = tf.keras.layers.Flatten()(x)
encoded = tf.keras.layers.Dense(8192, activation='leaky_relu')(x)

x = tf.keras.layers.Dense(32 * 32 * 512, activation='leaky_relu')(encoded)
x = tf.keras.layers.Reshape((32, 32, 512))(x)

x = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='leaky_relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='leaky_relu', strides=2, padding='same')(x)
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='leaky_relu', strides=2, padding='same')(x)

decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')

autoencoder.fit(train_dataset, epochs=15, shuffle=True, validation_data=validation_dataset, verbose=2)

decoded_imgs = autoencoder.predict(validation_dataset.take(1))

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(decoded_imgs[i])
    plt.axis('off')

    original_img = next(iter(validation_dataset))
    plt.subplot(2, 10, 10 + i + 1)
    plt.imshow(original_img[0][i])
    plt.axis('off')

plt.suptitle("Top: Reconstruction | Bottom: Original", fontsize=14)

plt.tight_layout()
plt.savefig("reconstruction.png")
plt.close()