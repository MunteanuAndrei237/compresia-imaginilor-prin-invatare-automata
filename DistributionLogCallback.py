import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

class DistributionLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, split, autoencoder, latent_layer_name="flatten",  quantization_scale = 192):
        super().__init__()
        self.split = split
        self.mean = None
        self.std = None
        self.quantization_scale = quantization_scale
        self.encoder = tf.keras.Model(
            inputs=autoencoder.input,
            outputs=autoencoder.get_layer(latent_layer_name).output
        )

    def get_distribution_stats(self):
        return self.mean, self.std

    def on_epoch_end(self, epoch, logs=None):
        # every 10 epochs take a batch and see how distribution of latents evolves
        if epoch % 10 == 0:
            for x_batch in self.split:
                encoded = self.encoder(x_batch, training=False)
                latents = tf.reshape(encoded, [encoded.shape[0], -1])
                latents = latents.numpy()
                break

            self.mean = np.mean(latents)
            self.std = np.std(latents)

            values = latents.flatten()
            plt.hist(values, bins=self.quantization_scale, density=True)

            x = np.linspace(min(values), max(values), self.quantization_scale)
            pdf = laplace.pdf(x, loc=self.mean, scale=self.std / np.sqrt(2))
            plt.plot(x, pdf, color='red', label='Laplace distribution')

            plt.xlabel('Latents')
            plt.grid(True)
            plt.legend()
            plt.show()
