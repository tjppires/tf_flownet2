## FlowNet2 (TensorFlow)

This repo includes the code for running FlowNet 2 on Tensorflow. It does NOT support training. The goal of this code is just to be able to quickly run and use FlowNet 2 using pure Tensorflow, WITHOUT any custom ops.
The code here is derived from [this repo](https://github.com/vt-vl-lab/tf_flownet2), which in turn is derived from [this](https://github.com/sampepose/flownet2-tf). It can deal with inputs of arbitrary sizes.

The main differences between the code here and the one in the repos above are:
* No dependencies on custom ops. All the code here is pure Python code + Tensorflow code. Just install the requirements, download the checkpoints and you should be good to go.
* No training code. The code in this repo is only meant for running the model. If you are interested in training a FlowNet 2 model, it should be relatively simple to add the training code from the repos above back.

### Environment

This code has been tested with Python 3.7.5 and TensorFlow 2.1.0.

### Installation

Clone the repo, install the requirements, and download the weights:

```
git clone https://github.com/vt-vl-lab/tf_flownet2.git
cd tf_flownet2
pip install -r requirements.txt
checkpoints/download.sh
```

Running the model should be easy, just check the available demo:

```
python demo.py
```

### Reference
[1] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, T. Brox
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks,
IEEE Conference in Computer Vision and Pattern Recognition (CVPR), 2017.

### Acknowledgments
As noted above, this code is based on [vt-vl-lab/tf_flownet2](https://github.com/vt-vl-lab/tf_flownet2) and [sampepose/flownet2-tf](https://github.com/sampepose/flownet2-tf).
