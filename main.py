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


def train(learning_rate, beta1, l1_lambda, max_epochs):
    xs, ys = get_input("train")
    print("load input successfully")
    print("input x shape is {}".format(xs.shape))
    print("input y shape is {}".format(ys.shape))

    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 256, 256, 3], name="y-input")

    G_sample = models.generator(x)

    logits_fake = models.con_discriminator(x, G_sample)
    logits_real = models.con_discriminator(x, y_)

    # get loss
    D_loss, G_loss_gan = loss.gan_loss(logits_fake=logits_fake, logits_real=logits_real)
    G_loss = G_loss_gan + l1_lambda * loss.l1_loss(y_, G_sample)
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("summary/")

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

    # iterations
    for epoch in range(max_epochs):
        print("Epoch: {}".format(epoch))
        for it in range(400):
            mask = np.random.choice(400, 1)
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: xs[mask], y_: ys[mask]})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: xs[mask], y_: ys[mask]})
            summary, _, G_loss_curr = sess.run([merged, G_train_step, G_loss], feed_dict={x: xs[mask], y_: ys[mask]})
            train_writer.add_summary(summary)

            if it % 50 == 0:
                print("iter {}: D_loss: {}, G_loss: {}".format(it, D_loss_curr, G_loss_curr))

        # save 5 sample images for each epoch
        for i in range(5):
            samples = sess.run(G_sample, feed_dict={x: xs[i:i+1], y_: ys[i:i+1]})
            img = 255 * (np.array(samples[0] + 1) / 2)
            im = Image.fromarray(np.uint8(img))
            im.save("samples/epoch{}_{}.jpg".format(epoch, i))
            
    return 0


def main(a):

    # input data
    # pre process data
    train(a.lr, a.beta1, a.l1_lambda, a.max_epochs)
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
    parser.add_argument("--summary_freq")
    parser.add_argument("--progress_freq")
    parser.add_argument("--trace_freq")
    parser.add_argument("--display_freq")
    parser.add_argument("--save_freq")

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
