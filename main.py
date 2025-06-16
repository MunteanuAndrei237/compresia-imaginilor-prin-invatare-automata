import tensorflow as tf
import tensorflow_datasets as tfds

from RateDistortionTrainer import RateDistortionTrainer
from reconstruction import reconstruct
from autoencoder import autoencoder
from DistributionLogCallback import DistributionLogCallback

#hyperparameters
BATCH_SIZE = 8
EXAMPLE_IMAGES = 8
EPOCHS = 200
LAMBDA_RATE_DISTORTION = 50
LEARN_RATE = 3.16e-4
IMAGE_SIZE = 256
ADD_NOISE = True
NOISE_RANGE = 0.25
LATENT_LAYER_NAME = "flatten"
QUANTIZATION_SCALE = 192
DATASET_NAME = "oxford_flowers102"

dataset, info = tfds.load(DATASET_NAME, with_info=True)
train_split = dataset['test'].take(6149) # we use the test set as training data because it has the most images
validation_split = dataset['train'].take(320)

def preprocess_image(example):
    raw_image = example['image']
    resized_image = tf.image.resize(raw_image, (IMAGE_SIZE, IMAGE_SIZE))
    normalized_image = tf.cast(resized_image, tf.float32) / 255.0
    return normalized_image

train_split = train_split.map(preprocess_image).batch(BATCH_SIZE)
validation_split = validation_split.map(preprocess_image).batch(BATCH_SIZE)

autoencoder = autoencoder(image_size = IMAGE_SIZE)
autoencoder.summary()
# either load pretrained weights or train the model
# autoencoder.load_weights("pretrained_model_weights/weights_lambda_50_epochs_200.h5")

distribution_log = DistributionLogCallback(split=validation_split, autoencoder=autoencoder, latent_layer_name=LATENT_LAYER_NAME, quantization_scale=QUANTIZATION_SCALE)

model = RateDistortionTrainer(autoencoder, lambda_rate_distortion=LAMBDA_RATE_DISTORTION, add_noise=ADD_NOISE, noise_range=NOISE_RANGE)
model.compile(optimizer=tf.keras.optimizers.Adam(LEARN_RATE))
model.fit(
    train_split,
    epochs=EPOCHS,
    validation_data=validation_split,
    callbacks=[distribution_log]
)

autoencoder.save_weights("pretrained_model_weights/weights_example.h5")

reconstruct(split=validation_split, autoencoder=autoencoder, latent_layer_name=LATENT_LAYER_NAME, lambda_rate_distortion=LAMBDA_RATE_DISTORTION, example_images=EXAMPLE_IMAGES, quantization_scale=QUANTIZATION_SCALE)