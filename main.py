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
    dataset = None
    if model == "train":
        dataset = np.zeros((400, 256, 512, 3))
        for i in range(1, 401):
            img = Image.open("dataset/{}/{}.jpg".format(model, i))
            dataset[i - 1] = np.array(img)

    if model == ("test" or "val"):
        dataset = np.zeros((100, 256, 512, 3))
        for i in range(1, 101):
            img = Image.open("dataset/{}/{}.jpg".format(model, i))
            dataset[i - 1] = np.array(img)

    # rescale [0, 255] to [-1, 1]
    dataset = dataset / 255 * 2 - 1
    y = dataset[:, :, :256, :]
    x = dataset[:, :, 256:, :]
    return x, y


def save_sample_img(samples, step, mode="train"):
    for i in range(5):
        # rescale [-1, 1] to [0, 255]
        img = 255 * (np.array(samples[i] + 1) / 2)
        im = Image.fromarray(np.uint8(img))
        im.save("samples/{}_step{}_{}.jpg".format(mode, step, i))


def get_solver(learning_rate=2e-4, beta1=0.5):
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    return D_solver, G_solver


def train(learning_rate, beta1, l1_lambda, max_epochs,
          summary_freq, display_freq, save_freq):
    xs_train, ys_train = get_input("train")
    xs_val, ys_val = get_input("val")
    print("load input successfully")
    print("input x shape is {}".format(xs_train.shape))
    print("input y shape is {}".format(ys_train.shape))

    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 256, 256, 3], name="y-input")

    G_sample = models.generator(x)

    logits_fake = models.con_discriminator(x, G_sample)
    logits_real = models.con_discriminator(x, y_)

    # get loss
    D_loss, G_loss_gan = loss.gan_loss(logits_fake=logits_fake, logits_real=logits_real)
    l1_loss = loss.l1_loss(y_, G_sample)
    with tf.variable_scope("G_loss"):
        G_loss = G_loss_gan + l1_lambda * l1_loss
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("summary/", sess.graph)

    # get weights list
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    # get solver
    D_solver, G_solver = get_solver(learning_rate=learning_rate, beta1=beta1)

    # get training steps
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

    # init
    sess.run(tf.global_variables_initializer())

    # get saver
    saver = tf.train.Saver()

    # iterations
    for step in range(max_epochs * 400):
        if step % 400 == 0:
            print("Epoch: {}".format(step / 400))

        mask = np.random.choice(400, 1)
        _, D_loss_curr = sess.run([D_train_step, D_loss],
                                  feed_dict={x: xs_train[mask], y_: ys_train[mask]})
        _, G_loss_curr = sess.run([G_train_step, G_loss],
                                  feed_dict={x: xs_train[mask], y_: ys_train[mask]})
        _, G_loss_curr = sess.run([G_train_step, G_loss],
                                  feed_dict={x: xs_train[mask], y_: ys_train[mask]})

        if step % display_freq == 0:
            print("iter {}: D_loss: {}, G_loss: {}".format(step, D_loss_curr, G_loss_curr))

        # save summary and checkpoint
        if step % summary_freq == 0:
            summary = sess.run(merged, feed_dict={x: xs_train, y_: ys_train})
            train_writer.add_summary(summary)
            saver.save(sess, summary, global_step=step)

        # save 5 sample images
        if step % save_freq == 0:
            samples_train = sess.run(G_sample, feed_dict={x: xs_train[0:5], y_: ys_train[0:5]})
            save_sample_img(samples_train, step=step, mode="train")
            samples_val = sess.run(G_sample, feed_dict={x: xs_val[0:5], y_: ys_val[0:5]})
            save_sample_img(samples_val, step=step, mode="val")

    return 0


def test(checkpoint_dir="summary/"):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_dir)

    xs, ys = get_input("test")

    sample = sess.run()

    return 0


def main(a):

    # input data
    # pre process data
    if a.mode == "train":
        train(learning_rate=a.lr, beta1=a.beta1, l1_lambda=a.l1_lambda, max_epochs=a.max_epochs,
              display_freq=a.display_freq, save_freq=a.save_freq, summary_freq=a.summary_freq)

    if a.mode == "test":
        test()
    # train
    # iter

    return 0


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--seed")
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--max_epochs", type=int, default=200, help="max number of epochs")
    parser.add_argument("--summary_freq", type=int, default=200, help="summary frequency")
    parser.add_argument("--display_freq", type=int, default=50, help="display frequency")
    parser.add_argument("--save_freq", type=int, default=400, help="save frequency")

    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_lambda", type=float, default=100.0, help="weight for L1 term")

    # export options
    parser.add_argument("--output_filetype", default="jpg", choices=["png", "jpg"])
    a = parser.parse_args()
    print(a.mode)

    EPS = 1e-12
    CROP_SIZE = 256

    main(a)
