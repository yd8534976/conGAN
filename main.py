import tensorflow as tf
import numpy as np
from PIL import Image
from scipy import misc

import models
import loss

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_input(model="train"):
    dataset = np.zeros((400, 256, 512, 3))
    for i in range(1, 401):
        img = Image.open("dataset/" + model + "/" + str(i) + ".jpg")
        dataset[i - 1] = np.array(img)
    dataset = dataset / 255 * 2 - 1
    y = dataset[:, :, :256, :]
    x = dataset[:, :, 256:, :]
    return x, y


def get_solver(learning_rate=2e-4, beta1=0.5):
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    return D_solver, G_solver


def sample_noise(shape):
    return tf.random_uniform(shape, -1, 1)


def train():
    xs, ys = get_input()
    print("load input successfully")
    print("input x shape is {}".format(xs.shape))
    print("input y shape is {}".format(ys.shape))

    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 256, 256, 3], name="y-input")
    z = sample_noise((1, 256, 256, 3))
    G_sample = models.generator(x, z)

    logits_fake = models.con_discriminator(x, G_sample, name='D1')
    logits_real = models.con_discriminator(x, y_, name='D2')

    # get loss
    D_loss, G_loss_gan = loss.lsgan_loss(logits_fake=logits_fake, logits_real=logits_real)
    G_loss = G_loss_gan + 100 * loss.l1_loss(y_, G_sample)
    # get solver
    D_solver, G_solver = get_solver()

    # get training steps
    D_train_step = D_solver.minimize(D_loss)
    G_train_step = G_solver.minimize(G_loss)

    # init
    sess.run(tf.global_variables_initializer())

    # iterations
    for epoch in range(300):
        print("Epoch: {}".format(epoch))
        for it in range(400):
            mask = np.random.choice(400, 1)
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: xs[mask], y_: ys[mask]})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: xs[mask], y_: ys[mask]})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: xs[mask], y_: ys[mask]})

            if it % 50 == 0:
                print("iter {}: D_loss: {}, G_loss: {}".format(it, D_loss_curr, G_loss_curr))
        samples = sess.run(G_sample, feed_dict={x: xs[0:1], y_: ys[0:1]})

        img = 255 * (np.array(samples[0]) + 1) / 2
        im = Image.fromarray(np.uint8(img))
        im.save("test/generated_sample" + str(epoch) + ".jpg")
    return 0


def main():

    # input data
    # pre process data
    train()
    # train
    # iter

    return 0


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--mode")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed")
    parser.add_argument("--checkpoint")

    parser.add_argument("--max_steps")
    parser.add_argument("--max_epochs")
    parser.add_argument("--summary_freq")
    parser.add_argument("--progress_freq")
    parser.add_argument("--trace_freq")
    parser.add_argument("--display_freq")
    parser.add_argument("--save_freq")

    parser.add_argument("--aspect_ratio")
    parser.add_argument("--lab_colorization")
    parser.add_argument("--batch_size")
    parser.add_argument("--which_direction")
    parser.add_argument("--ngf")
    parser.add_argument("--ndf")
    parser.add_argument("--scale_size")
    parser.add_argument("--flip")
    parser.add_argument("--no_flip")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr")
    parser.add_argument("--beta1")
    parser.add_argument("--l1_weight")
    parser.add_argument("--gan_weight")

    # export options
    parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
    a = parser.parse_args()

    EPS = 1e-12
    CROP_SIZE = 256

    main()
