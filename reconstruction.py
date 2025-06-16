import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from arithmetic import ArithmeticEncoder, ArithmeticDecoder
from skimage.metrics import structural_similarity as ssim

def psnr(mse, max_pixel=1.0):
    return 10 * np.log10((max_pixel ** 2) / mse)

def reconstruct(split, autoencoder, latent_layer_name ="flatten", lambda_rate_distortion = 50, example_images=8, quantization_scale=192):

    # get latent layer index
    latent_layer_index = autoencoder.layers.index(autoencoder.get_layer(latent_layer_name))

    # reconstruct the encoder
    encoder_input = autoencoder.input
    encoded_latent = autoencoder.layers[latent_layer_index].output
    encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoded_latent)

    # reconstruct the decoder
    reshape_layer = autoencoder.layers[latent_layer_index + 1]
    decoder_input = tf.keras.Input(shape=(int(tf.reduce_prod( reshape_layer.target_shape)),))

    layer = reshape_layer(decoder_input)
    for i in range(latent_layer_index + 2, len(autoencoder.layers)):
        layer = autoencoder.layers[i](layer)
    decoder_model = tf.keras.Model(inputs=decoder_input, outputs=layer)

    sample_batch = next(iter(split))
    original_images = sample_batch[:example_images]

    # results lists
    reconstructed_latents = []
    mse_list = []
    psnr_list = []
    ssim_list = []
    byte_lengths = []
    dictionary_sizes = []

    latents = encoder_model.predict(original_images)
    quantized_latents = np.round(latents * quantization_scale).astype(int)

    # arithmetic coding
    for i in range(len(quantized_latents)):
        q_latent = quantized_latents[i]
        min_symbol = q_latent.min()
        shifted = q_latent - min_symbol

        counts = np.bincount(shifted)
        cumulative_frequencies = np.concatenate(([0], np.cumsum(counts + 1)))
        dictionary_sizes.append(cumulative_frequencies.nbytes)

        encoder = ArithmeticEncoder(32, cumulative_frequencies)
        for symbol in shifted:
            encoder.write(symbol)
        encoder.finish()

        encoded_bytes = encoder.get_encoded_data()

        decoder = ArithmeticDecoder(32, cumulative_frequencies, encoded_bytes)
        decoded = np.array([decoder.read() for _ in range(len(shifted))])
        decoded_original = decoded + min_symbol
        dequantized = decoded_original / quantization_scale
        reconstructed_latents.append(dequantized)
        byte_lengths.append(len(encoded_bytes))

    reconstructed_latents = np.stack(reconstructed_latents)
    reconstructed_images = decoder_model.predict(reconstructed_latents)

    # store results
    for i in range(len(original_images)):
        orig_img = original_images[i].numpy()
        recon_img = reconstructed_images[i]
        mse_list.append(np.mean((orig_img - recon_img) ** 2))
        psnr_list.append(psnr(np.mean((orig_img - recon_img) ** 2)))
        ssim_list.append(ssim(orig_img, recon_img, data_range=1.0, channel_axis=-1))

    # plot results
    plt.figure(figsize=(25, 6))
    for i in range(example_images):

        plt.subplot(2, example_images, i + 1)
        plt.imshow(original_images[i])
        plt.axis("off")

        plt.subplot(2, example_images, example_images + i + 1)
        plt.imshow(reconstructed_images[i])
        plt.axis("off")

        plt.title("MSE: " + str(round(mse_list[i], 4)) +
                  "\nPSNR: " + str(round(psnr_list[i], 2)) + " dB"
                  + "\nSSIM: " + str(round(ssim_list[i], 4)) +"\nBpp: "
                  + str(round((byte_lengths[i] + dictionary_sizes[i]) * 8 / 65536, 3)),
                  fontsize=17
        )

    plt.suptitle(
        "Pe primul rând se află pozele originale, iar pe al doilea rând se află pozele reconstruite. Lambda: " + str(lambda_rate_distortion) +
        "\nMSE mediu: " + str(round(np.mean(mse_list), 4)) + "  " +
        "PSNR mediu: " + str(round(np.mean(psnr_list), 2)) + " dB  " +
        "SSIM mediu: " + str(round(np.mean(ssim_list), 4)) + "  " +
        "Bpp mediu: " + str(round((np.mean(byte_lengths) + np.mean(dictionary_sizes)) * 8 / 65536, 3)),
        fontsize=24
    )

    plt.tight_layout()
    plt.show()
