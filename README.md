# conditional GAN (pix2pix-tensorflow)

[[Paper]](https://arxiv.org/pdf/1611.07004v1.pdf) [[Torch ver.]](https://github.com/phillipi/pix2pix)

![examples]
(./examples.jpg)

## Setup
### Prerequisites
- Linux
- GPU
- tensorflow-1.4
- tensorboard (optional)
### Getting Started

- Clone this repo:
````
git clone https://github.com/yd8534976/conGAN.git
cd conGAN
````
- Download the dataset:
````
python ./tools/download-dataset.py facades
mv facades dataset
````
- Train mode
````
python main.py --mode="train"
````
- Test mode
````
python main.py --mode="test"
````
### Visualization
````
tensorboard --logdir=summary/
````
### Citation
If you use this code for your research, cite paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)
````
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}
````
