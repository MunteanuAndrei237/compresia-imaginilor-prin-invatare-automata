import tensorflow as tf

class RateDistortionTrainer(tf.keras.Model):
    def __init__(self, autoencoder, lambda_rate_distortion=50, add_noise=True, noise_range=0.25):
        super().__init__()
        self.optimizer = None
        self.autoencoder = autoencoder
        self.lambda_rate_distortion = lambda_rate_distortion
        self.add_noise = add_noise
        self.noise_range = noise_range

        self.mse = tf.keras.metrics.Mean(name="mse")
        self.variance = tf.keras.metrics.Mean(name="variance")
        self.total_loss = tf.keras.metrics.Mean(name="total_loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def compute_variance(self, latents):
        return tf.math.reduce_variance(latents, axis=1)

    def train_step(self, data):
        input_image = data

        with tf.GradientTape() as tape:
            latent, reconstructed = self.autoencoder(input_image, training=True)
            mse = tf.keras.losses.MeanSquaredError()(input_image, reconstructed)

            if self.add_noise:
                latent = latent + tf.random.uniform(tf.shape(latent), minval=-self.noise_range, maxval=self.noise_range)

            variance = self.compute_variance(latent)
            total_loss = variance + self.lambda_rate_distortion * mse

        gradients = tape.gradient(total_loss, self.autoencoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))

        self.mse.update_state(mse)
        self.variance.update_state(variance)
        self.total_loss.update_state(total_loss)

        return {
            "mse": self.mse.result(),
            "variance": self.variance.result(),
            "total_loss": self.total_loss.result()
        }

    def test_step(self, data):
        input_image = data
        latent, reconstructed = self.autoencoder(input_image, training=False)

        mse = tf.keras.losses.MeanSquaredError()(input_image, reconstructed)
        variance = self.compute_variance(latent)
        total_loss = variance + self.lambda_rate_distortion * mse

        self.mse.update_state(mse)
        self.variance.update_state(variance)
        self.total_loss.update_state(total_loss)

        return {
            "mse": self.mse.result(),
            "variance": self.variance.result(),
            "total_loss": self.total_loss.result()
        }
