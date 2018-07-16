import tensorflow as tf
import numpy as np
from PIL import Image

import models
import loss

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

NUM_TRAIN_IMAGES = 400
NUM_TEST_IMAGES = 100


def get_input(mode="train"):
    dataset = np.zeros((100, 256, 512, 3))
    if mode == "train":
        dataset = np.zeros((NUM_TRAIN_IMAGES, 256, 512, 3))
        for i in range(1, 401):
            img = Image.open("dataset/{}/{}.jpg".format(mode, i))
            dataset[i - 1] = np.array(img)
    else:
        dataset = np.zeros((NUM_TEST_IMAGES, 256, 512, 3))
        for i in range(1, 101):
            img = Image.open("dataset/{}/{}.jpg".format(mode, i))
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
        im.save("samples/{}/step{}_{}.jpg".format(mode, step, i))


def get_solver(learning_rate=2e-4, beta1=0.5):
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    return D_solver, G_solver


def run_model(mode, learning_rate=2e-4, beta1=0.5, l1_lambda=100, max_epochs=200,
              summary_freq=200, display_freq=50, save_freq=400,
              checkpoint_dir="summary/conGAN.ckpt"):
    if mode == "train":
        xs_train, ys_train = get_input("train")
        xs_val, ys_val = get_input("val")
        print("load train data successfully")
        print("input x shape is {}".format(xs_train.shape))
        print("input y shape is {}".format(ys_train.shape))
    else:
        xs_test, ys_test = get_input("test")
        print("load test data successfully")
        print("input x shape is {}".format(xs_test.shape))
        print("input y shape is {}".format(ys_test.shape))

    # build model
    # -----------
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

    # get weights list
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    # get solver
    D_solver, G_solver = get_solver(learning_rate=learning_rate, beta1=beta1)

    # get training steps

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
        G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    # -----------

    # get session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # get saver
    saver = tf.train.Saver()

    # training phase
    if mode == "train":
        train_writer = tf.summary.FileWriter("summary/", sess.graph)
        # init
        sess.run(tf.global_variables_initializer())

        # iterations
        for step in range(max_epochs * NUM_TRAIN_IMAGES):
            if step % NUM_TRAIN_IMAGES == 0:
                print("Epoch: {}".format(step / NUM_TRAIN_IMAGES))

            mask = np.random.choice(NUM_TRAIN_IMAGES, 1)
            _, D_loss_curr = sess.run([D_train_step, D_loss],
                                      feed_dict={x: xs_train[mask], y_: ys_train[mask]})
            _, G_loss_curr = sess.run([G_train_step, G_loss],
                                      feed_dict={x: xs_train[mask], y_: ys_train[mask]})
            _, G_loss_curr = sess.run([G_train_step, G_loss],
                                      feed_dict={x: xs_train[mask], y_: ys_train[mask]})

            if step % display_freq == 0:
                print("step {}: D_loss: {}, G_loss: {}".format(step, D_loss_curr, G_loss_curr))

            # save summary and checkpoint
            if step % summary_freq == 0:
                mask = np.random.choice(NUM_TRAIN_IMAGES, 30)
                summary = sess.run(merged, feed_dict={x: xs_train[mask], y_: ys_train[mask]})
                train_writer.add_summary(summary)
                saver.save(sess, checkpoint_dir)

            # save 5 sample images
            if step % save_freq == 0:
                samples_train = sess.run(G_sample, feed_dict={x: xs_train[0:5], y_: ys_train[0:5]})
                save_sample_img(samples_train, step=step, mode="train")
                samples_val = sess.run(G_sample, feed_dict={x: xs_val[0:5], y_: ys_val[0:5]})
                save_sample_img(samples_val, step=step, mode="val")

    # testing phase
    if mode == "test":
        saver.restore(sess, checkpoint_dir)
        for i in range(20):
            samples_test = sess.run(G_sample, feed_dict={x: xs_test[5*i:5*(i+1)], y_: ys_test[5*i:5*(i+1)]})
            save_sample_img(samples_test, step=i, mode="test")

    # close sess
    sess.close()

    return 0


def main(a):
    if a.mode == "train":
        run_model(mode="train", learning_rate=a.lr, beta1=a.beta1, l1_lambda=a.l1_lambda, max_epochs=a.max_epochs,
                  display_freq=a.display_freq, save_freq=a.save_freq, summary_freq=a.summary_freq,
                  checkpoint_dir=a.checkpoint_dir)

    if a.mode == "test":
        run_model(mode="test")

    return 0


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--seed")
    parser.add_argument("--checkpoint_dir", default="summary/conGAN.ckpt")

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

    main(a)
