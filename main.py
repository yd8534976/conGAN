import tensorflow as tf
import numpy as np

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    # arg parse

    # pre process data

    # train
    # iter

    return 0


if __name__ == "__main__":
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
    parser.add_argument("--output_filetype")
    a = parser.parse_args()
    print(main())
