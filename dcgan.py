"""
DCGAN implementation (based on the Keras DCGAN tutorial)

Tensorflow 2.1/Keras 

(should work from TF2.0)


experiments to train with an autoencoder on 512x512 images


"""


import os
import sys
import argparse
import numpy as np

from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint

import cv2
import imutils
import datetime
import glob
import imageio
import matplotlib.pyplot as plt

import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorboardX import SummaryWriter
import tqdm
import yaml
from easydict import EasyDict

import wgan

# fixed config args
with open('cfg/main.yaml', 'r') as f:
    cargs = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))


# def gen_layers(num_filter, )

def get_time_now():
    now = datetime.datetime.now()
    now = now.strftime("%d-%m-%Y_%H-%M-%S")
    return now


def add_layers(model, filter_size, assert_output=None):
    c2dt = layers.Conv2DTranspose(filter_size, (5, 5), strides=(2, 2),
                                  padding='same', use_bias=False)
    bn = layers.BatchNormalization()
    lrelu = layers.LeakyReLU()

    model.add(c2dt)
    if assert_output is not None:
        assert model.output_shape == assert_output
    model.add(bn)
    model.add(lrelu)


def make_encoder_model(noise_dim):
    model = tf.keras.Sequential()
    # model.add(layers.Input(shape=(512,512,3)))

    model.add(layers.Conv2D(
        16, (5, 5), strides=(2, 2),
        input_shape=(512, 512, 3),
        padding='same', use_bias=False))
    assert model.output_shape == (None, 256, 256, 16), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(
        32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 32), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Conv2D(
    #     128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 32, 32, 128), model.output_shape
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2D(
    #     256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 16, 16, 256), model.output_shape
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2D(
    #     512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 8, 8, 512), model.output_shape
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    # assert model.output_shape == (None, 32*32*128)
    # model.add(layers.Dense(noise_dim))
    return model


def make_generator_model(noise_dim):
    model = tf.keras.Sequential()
    # model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(noise_dim,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    model.add(layers.Input(shape=(noise_dim,)))

    model.add(layers.Reshape((64, 64, 64)))
    # Note: None is the batch size
    # assert model.output_shape == (None, 32, 32, 128)
    assert model.output_shape == (None, 64, 64, 64)

    # model.add(layers.Conv2DTranspose(
    #     512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 8, 8, 512)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2DTranspose(
    #     256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 16, 16, 256)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2DTranspose(
    #     128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 32, 32, 128)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # model.add(layers.Conv2DTranspose(
    #     64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 64, 64, 64)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 256, 256, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 512, 512, 3)

    return model


def make_autoencoder_model(noise_dim):
    pass


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[512, 512, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(100))
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def autoencoder_loss(real_image, generated_image):
    mse = tf.keras.losses.MeanSquaredError()
    total_mse = mse(real_image[:, :, :, 0], generated_image[:, :, :, 0]) + \
        mse(real_image[:, :, :, 1], generated_image[:, :, :, 1]) + \
        mse(real_image[:, :, :, 2], generated_image[:, :, :, 2])
    return total_mse

# def show_test_image():

#     generator = make_generator_model()

#     noise = tf.random.normal([1, 100])
#     generated_image = generator(noise, training=False)

#     plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#     # plt.show()

#     discriminator = make_discriminator_model()
#     decision = discriminator(generated_image)
#     print(decision)


def bgr2rgb(img):
    return img[:, :, ::-1]


def autoencode_ds(sample_it, autoencoder, img_shape, n_sample=16):

    for i in range(n_sample):
        img, _ = sample_it.next()
        generated_image = autoencoder(img, training=False)
        image = generated_image[0].numpy() * 127.5 + 127.5
        img_ae = bgr2rgb(image)
        cv2.imwrite("generated/image_{}.png".format(i), img_ae)


def do(args):
    # (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    path = cargs.ds_path
    nb_image = 15
    # train_images = np.zeros((nb_image, 28, 28))
    # for i in range(nb_image):
    #     fname = path + "{}.jpg".format(i+1)
    #     img = cv2.imread(fname, 0)
    #     img = cv2.resize(img, (28,28))
    #     cv2.imwrite("orig/image_{}.png".format(i), img)
    #     train_images[i] = img

    BUFFER_SIZE = cargs.buffer_size
    BATCH_SIZE = cargs.batch_size
    EPOCHS = cargs.epochs
    start_epoch = args.startepoch
    # noise_dim = 100
    # noise_dim = 8*8*512
    noise_dim = 64 * 64 * 64

    num_examples_to_generate = cargs.num_ex_to_gen
    img_shape = cargs.img_shape_w, cargs.img_shape_h

    # 940
    train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                       horizontal_flip=True, vertical_flip=True)

    orig_datagen = ImageDataGenerator()

    train_it = train_datagen.flow_from_directory(path, batch_size=BATCH_SIZE, target_size=img_shape,
                                                 color_mode="rgb", shuffle=True)

    orig_it = orig_datagen.flow_from_directory(path, batch_size=BATCH_SIZE, target_size=img_shape,
                                               color_mode="rgb")
    sample_it = orig_datagen.flow_from_directory(path, batch_size=1, target_size=img_shape,
                                                 color_mode="rgb")

    for i in range(nb_image):
        fname = path + "{}.jpg".format(i + 1)
        img, _ = orig_it.next()
        img = bgr2rgb(img[0])
        # breakpoint()
        cv2.imwrite("orig/image_{}.png".format(i), img)

    # breakpoint()

    # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    # train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    # Batch and shuffle the data
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    # mse = tf.keras.losses.MeanSquaredError()

    generator = make_generator_model(noise_dim)
    encoder = make_encoder_model(noise_dim)

    inp = layers.Input(shape=(512, 512, 3))
    # breakpoint()
    # inp1 = encoder.layers[0](inp)
    # encoded = generator.layers[0](encoder(inp))
    outp = generator(encoder(inp))

    autoencoder = tf.keras.Model(inputs=[inp], outputs=[outp])

    discriminator = make_discriminator_model()

    # autoencoder = tf.keras.Model(encoder.layers[0], generator(encoder.layers[0]))

    generator_optimizer = tf.keras.optimizers.Adam(cargs.lr_gen)
    discriminator_optimizer = tf.keras.optimizers.Adam(cargs.lr_disc)
    ae_opt = tf.keras.optimizers.Adam(cargs.lr_ae)

    wgan_critic_opt = tf.keras.optimizers.RMSProp(lr=cargs.lr_critic)
    wgan_gan_opt = tf.keras.optimizers.RMSProp(lr=cargs.lr_critic)

    writer = SummaryWriter(logdir=cargs.log_path, max_queue=1)

    checkpoint_dir = cargs.checkpoints_path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    # breakpoint()

    if args.restore:
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    n_sample = len(train_it)

    now = get_time_now()

    @tf.function
    def train_step_gan(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        # tf.reshape(noise, (BATCH_SIZE, ))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output, cross_entropy)
            disc_loss = discriminator_loss(
                real_output, fake_output, cross_entropy)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

    @tf.function
    def train_step_ae(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as ae_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # latent_images = encoder(images, training=True)
            generated_images = autoencoder(images, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            ae_loss = autoencoder_loss(images, generated_images)

            gen_loss = generator_loss(fake_output, cross_entropy)
            disc_loss = discriminator_loss(real_output, fake_output,
                                           cross_entropy)

        # trainable vars
        ae_vars = autoencoder.trainable_variables
        gen_vars = generator.trainable_variables
        disc_vars = discriminator.trainable_variables

        # GRADIENTS
        grad_ae = ae_tape.gradient(ae_loss, ae_vars)
        grad_gen = gen_tape.gradient(gen_loss, gen_vars)
        grad_disc = disc_tape.gradient(disc_loss, disc_vars)
        # breakpoint()

        # AE UPDATE
        # ae_opt.apply_gradients(zip(grad_ae, ae_vars))

        # GEN UPDATE
        generator_optimizer.apply_gradients(zip(grad_gen, gen_vars))

        # DISC UPDATE
        discriminator_optimizer.apply_gradients(zip(grad_disc, disc_vars))

        return ae_loss, gen_loss, disc_loss

    def train(train_it, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            start = time.time()

            for ind in range(len(train_it)):
                image_batch, _ = train_it.next()

                image_batch = (image_batch - 127.5) / 127.5

                ae_loss, gen_loss, disc_loss = train_step_ae(image_batch)
                # train_step_gan(image_batch)

                print("batch {}     \r".format(ind), end="")

            end = time.time()
            ae_loss, gen_loss, disc_loss = ae_loss.numpy(), gen_loss.numpy(), disc_loss.numpy()
            loss_print = "{:.8} {:.8} {:.8}      \r".format(
                ae_loss, gen_loss, disc_loss)
            print('                 Time for epoch {} is {:.2} sec  '.format(
                epoch + 1, end - start) + loss_print, end="")

            writer.add_scalars("losses/{}_{}".format("dcgan_ae", now),
                               {
                'ae_loss': ae_loss.numpy(),
                'gen_loss': gen_loss.numpy(),
                'disc_loss': disc_loss.numpy(),
            }, epoch)

            # Save the model every 200 epochs
            if (epoch + 1) % 500 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                print(
                    "[Epoch {}] Checkpoint saved.                          ".format(epoch + 1))

            if (epoch + 1) % 20 == 0:
                predictions = generator(seed, training=False)
                for i in range(num_examples_to_generate):
                    # breakpoint()
                    img = predictions[i].numpy() * 127.5 + 127.5
                    img = bgr2rgb(img)
                    cv2.imwrite("from_noise/image_{}.png".format(i), img)

                # saves on disk example images generated by the AE
                autoencode_ds(sample_it, autoencoder, img_shape, n_sample=16)

    print("n_sample =", n_sample)
    train(train_it, EPOCHS, start_epoch=start_epoch)
    print("n_sample =", n_sample)
    writer.close()
    # predictions = generator(seed, training=False)

    # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                         epochs,
    #                         seed)

    # def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    # fig = plt.figure(figsize=(4,4))

    # for i in range(predictions.shape[0]):
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    #     plt.axis('off')

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

    # show_test_image()


def main():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--startepoch", default=0,
                        help="help")
    parser.add_argument("--restore", action="store_true",
                        help="help")
    args = parser.parse_args()
    print(args)

    # if args.arg:
    # do()
    do(args)


if __name__ == "__main__":
    sys.exit(main())
