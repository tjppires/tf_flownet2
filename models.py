"""Tensorflow implementation of FlowNet 2.0.

Code adapted from:
https://github.com/vt-vl-lab/tf_flownet2

This code removes the dependencies on custom ops, by replacing them by readily
available Tensorflow ops.
This code does NOT include the training code. It is only used for exporting a
SavedModel from the available checkpoints.
"""

import tensorflow.compat.v1 as tf
from tensorflow_addons import image as tfa_image
from tensorflow_addons import layers as tfa_layers
import tf_slim as slim

tf.disable_v2_behavior()

WEIGHT_DECAY = 0.0004

############ Utilities. ############


def leaky_relu(x, name="lrelu"):
  return tf.nn.leaky_relu(x, alpha=0.1, name=name)


def pad(tensor, num=1):
  """Pads tensor along the height and width dimensions with num 0s on each side."""
  return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, num=1):
  """Performs a crop.

  "padding" for a deconvolutional layer (conv2d transpose) removes
  padding from the output rather than adding it to the input.
  Args:
    tensor: Input tensor.
    num: The amount of padding to remove.

  Returns:
    Unpaded tensor or whatever. This code is bad.
  """
  batch, h, w, c = tensor.shape.as_list()
  return tf.slice(
      tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])


def correlation(input_a, input_b, kernel_size, max_displacement, stride_1,
                stride_2, padding):
  correlation_layer = tfa_layers.CorrelationCost(kernel_size, max_displacement,
                                                 stride_1, stride_2, padding,
                                                 "channels_last")
  return correlation_layer([input_a, input_b])


def flow_warp(image, flow):
  # Tensorflow addons uses a different notation for flow, hence the minus sign.
  return tfa_image.dense_image_warp(image, -flow)


def channel_norm(tensor):
  sq = tf.square(tensor)
  r_sum = tf.reduce_sum(sq, keep_dims=True, axis=3)
  return tf.sqrt(r_sum)


############ Models. ############


class FlowNetC:
  """Tensorflow implementation of FlowNetC."""

  def run(self, inputs, trainable=True):
    """Runs model."""
    _, height, width, _ = inputs["input_a"].shape.as_list()
    with tf.variable_scope("FlowNetC"):
      with slim.arg_scope(
          [slim.conv2d, slim.conv2d_transpose],
          # Only backprop this network if trainable.
          trainable=trainable,
          # He (aka MSRA) weight initialization.
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=leaky_relu,
          # We will do our own padding to match the original Caffe code.
          padding="VALID"):

        weights_regularizer = slim.l2_regularizer(WEIGHT_DECAY)
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=weights_regularizer):
          with slim.arg_scope([slim.conv2d], stride=2):
            conv_a_1 = slim.conv2d(
                pad(inputs["input_a"], 3), 64, 7, scope="conv1")
            conv_a_2 = slim.conv2d(pad(conv_a_1, 2), 128, 5, scope="conv2")
            conv_a_3 = slim.conv2d(pad(conv_a_2, 2), 256, 5, scope="conv3")

            conv_b_1 = slim.conv2d(
                pad(inputs["input_b"], 3), 64, 7, scope="conv1", reuse=True)
            conv_b_2 = slim.conv2d(
                pad(conv_b_1, 2), 128, 5, scope="conv2", reuse=True)
            conv_b_3 = slim.conv2d(
                pad(conv_b_2, 2), 256, 5, scope="conv3", reuse=True)

            # Compute cross correlation with leaky relu activation.
            cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
            cc_relu = leaky_relu(cc)

          # Combine cross correlation results with convolution of feature map A.
          net_a_conv = slim.conv2d(conv_a_3, 32, 1, scope="conv_redir")
          # Concatenate along the channels axis.
          net = tf.concat([net_a_conv, cc_relu], axis=3)

          conv3_1 = slim.conv2d(pad(net), 256, 3, scope="conv3_1")
          with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
            conv4 = slim.conv2d(pad(conv3_1), stride=2, scope="conv4")
            conv4_1 = slim.conv2d(pad(conv4), scope="conv4_1")
            conv5 = slim.conv2d(pad(conv4_1), stride=2, scope="conv5")
            conv5_1 = slim.conv2d(pad(conv5), scope="conv5_1")
          conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope="conv6")
          conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope="conv6_1")
          # START: Refinement Network.
          with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
            predict_flow6 = slim.conv2d(
                pad(conv6_1), 2, 3, scope="predict_flow6", activation_fn=None)
            deconv5 = antipad(
                slim.conv2d_transpose(
                    conv6_1, 512, 4, stride=2, scope="deconv5"))
            upsample_flow6to5 = antipad(
                slim.conv2d_transpose(
                    predict_flow6,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow6to5",
                    activation_fn=None))
            concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

            predict_flow5 = slim.conv2d(
                pad(concat5), 2, 3, scope="predict_flow5", activation_fn=None)
            deconv4 = antipad(
                slim.conv2d_transpose(
                    concat5, 256, 4, stride=2, scope="deconv4"))
            upsample_flow5to4 = antipad(
                slim.conv2d_transpose(
                    predict_flow5,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow5to4",
                    activation_fn=None))
            concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

            predict_flow4 = slim.conv2d(
                pad(concat4), 2, 3, scope="predict_flow4", activation_fn=None)
            deconv3 = antipad(
                slim.conv2d_transpose(
                    concat4, 128, 4, stride=2, scope="deconv3"))
            upsample_flow4to3 = antipad(
                slim.conv2d_transpose(
                    predict_flow4,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow4to3",
                    activation_fn=None))
            concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

            predict_flow3 = slim.conv2d(
                pad(concat3), 2, 3, scope="predict_flow3", activation_fn=None)
            deconv2 = antipad(
                slim.conv2d_transpose(
                    concat3, 64, 4, stride=2, scope="deconv2"))
            upsample_flow3to2 = antipad(
                slim.conv2d_transpose(
                    predict_flow3,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow3to2",
                    activation_fn=None))
            concat2 = tf.concat([conv_a_2, deconv2, upsample_flow3to2], axis=3)

            predict_flow2 = slim.conv2d(
                pad(concat2), 2, 3, scope="predict_flow2", activation_fn=None)
          # END: Refinement Network.

          flow = predict_flow2 * 20.0

          flow = tf.image.resize_bilinear(
              flow, tf.stack([height, width]), align_corners=True)

          return {
              "predict_flow6": predict_flow6,
              "predict_flow5": predict_flow5,
              "predict_flow4": predict_flow4,
              "predict_flow3": predict_flow3,
              "predict_flow2": predict_flow2,
              "flow": flow,
          }


class FlowNetS:
  """Tensorflow implementation of FlowNetS."""

  def run(self, inputs, trainable=True):
    """Runs model."""
    _, height, width, _ = inputs["input_a"].shape.as_list()
    with tf.variable_scope("FlowNetS"):
      if "warped" in inputs and "flow" in inputs and "brightness_error" in inputs:
        concat_inputs = tf.concat([
            inputs["input_a"], inputs["input_b"], inputs["warped"],
            inputs["flow"], inputs["brightness_error"]
        ],
                                  axis=3)
      else:
        concat_inputs = tf.concat([inputs["input_a"], inputs["input_b"]],
                                  axis=3)
      with slim.arg_scope(
          [slim.conv2d, slim.conv2d_transpose],
          # Only backprop this network if trainable.
          trainable=trainable,
          # He (aka MSRA) weight initialization.
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=leaky_relu,
          # We will do our own padding to match the original Caffe code.
          padding="VALID"):

        weights_regularizer = slim.l2_regularizer(WEIGHT_DECAY)
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=weights_regularizer):
          with slim.arg_scope([slim.conv2d], stride=2):
            conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope="conv1")
            conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope="conv2")
            conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope="conv3")

          conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope="conv3_1")
          with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
            conv4 = slim.conv2d(pad(conv3_1), stride=2, scope="conv4")
            conv4_1 = slim.conv2d(pad(conv4), scope="conv4_1")
            conv5 = slim.conv2d(pad(conv4_1), stride=2, scope="conv5")
            conv5_1 = slim.conv2d(pad(conv5), scope="conv5_1")
          conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope="conv6")
          conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope="conv6_1")
          # START: Refinement Network.
          with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
            predict_flow6 = slim.conv2d(
                pad(conv6_1), 2, 3, scope="predict_flow6", activation_fn=None)
            deconv5 = antipad(
                slim.conv2d_transpose(
                    conv6_1, 512, 4, stride=2, scope="deconv5"))
            upsample_flow6to5 = antipad(
                slim.conv2d_transpose(
                    predict_flow6,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow6to5",
                    activation_fn=None))
            concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

            predict_flow5 = slim.conv2d(
                pad(concat5), 2, 3, scope="predict_flow5", activation_fn=None)
            deconv4 = antipad(
                slim.conv2d_transpose(
                    concat5, 256, 4, stride=2, scope="deconv4"))
            upsample_flow5to4 = antipad(
                slim.conv2d_transpose(
                    predict_flow5,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow5to4",
                    activation_fn=None))
            concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

            predict_flow4 = slim.conv2d(
                pad(concat4), 2, 3, scope="predict_flow4", activation_fn=None)
            deconv3 = antipad(
                slim.conv2d_transpose(
                    concat4, 128, 4, stride=2, scope="deconv3"))
            upsample_flow4to3 = antipad(
                slim.conv2d_transpose(
                    predict_flow4,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow4to3",
                    activation_fn=None))
            concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

            predict_flow3 = slim.conv2d(
                pad(concat3), 2, 3, scope="predict_flow3", activation_fn=None)
            deconv2 = antipad(
                slim.conv2d_transpose(
                    concat3, 64, 4, stride=2, scope="deconv2"))
            upsample_flow3to2 = antipad(
                slim.conv2d_transpose(
                    predict_flow3,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow3to2",
                    activation_fn=None))
            concat2 = tf.concat([conv_2, deconv2, upsample_flow3to2], axis=3)

            predict_flow2 = slim.conv2d(
                pad(concat2), 2, 3, scope="predict_flow2", activation_fn=None)
          # END: Refinement Network.

          flow = predict_flow2 * 20.0
          flow = tf.image.resize_bilinear(
              flow, tf.stack([height, width]), align_corners=True)

          return {
              "predict_flow6": predict_flow6,
              "predict_flow5": predict_flow5,
              "predict_flow4": predict_flow4,
              "predict_flow3": predict_flow3,
              "predict_flow2": predict_flow2,
              "flow": flow,
          }


class FlowNetSD:
  """Tensorflow implementation of FlowNetSD."""

  def run(self, inputs, trainable=True):
    """Runs model."""
    _, height, width, _ = inputs["input_a"].shape.as_list()
    with tf.variable_scope("FlowNetSD"):
      concat_inputs = tf.concat([inputs["input_a"], inputs["input_b"]], axis=3)
      with slim.arg_scope(
          [slim.conv2d, slim.conv2d_transpose],
          # Only backprop this network if trainable.
          trainable=trainable,
          # He (aka MSRA) weight initialization.
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=leaky_relu,
          # We will do our own padding to match the original Caffe code.
          padding="VALID"):

        weights_regularizer = slim.l2_regularizer(WEIGHT_DECAY)
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=weights_regularizer):
          conv0 = slim.conv2d(pad(concat_inputs), 64, 3, scope="conv0")
          conv1 = slim.conv2d(pad(conv0), 64, 3, stride=2, scope="conv1")
          conv1_1 = slim.conv2d(pad(conv1), 128, 3, scope="conv1_1")
          conv2 = slim.conv2d(pad(conv1_1), 128, 3, stride=2, scope="conv2")
          conv2_1 = slim.conv2d(pad(conv2), 128, 3, scope="conv2_1")
          conv3 = slim.conv2d(pad(conv2_1), 256, 3, stride=2, scope="conv3")
          conv3_1 = slim.conv2d(pad(conv3), 256, 3, scope="conv3_1")
          conv4 = slim.conv2d(pad(conv3_1), 512, 3, stride=2, scope="conv4")
          conv4_1 = slim.conv2d(pad(conv4), 512, 3, scope="conv4_1")
          conv5 = slim.conv2d(pad(conv4_1), 512, 3, stride=2, scope="conv5")
          conv5_1 = slim.conv2d(pad(conv5), 512, 3, scope="conv5_1")
          conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope="conv6")
          conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope="conv6_1")
          # START: Refinement Network.
          with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
            predict_flow6 = slim.conv2d(
                pad(conv6_1), 2, 3, scope="predict_flow6", activation_fn=None)
            deconv5 = antipad(
                slim.conv2d_transpose(
                    conv6_1, 512, 4, stride=2, scope="deconv5"))
            upsample_flow6to5 = antipad(
                slim.conv2d_transpose(
                    predict_flow6,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow6to5",
                    activation_fn=None))
            concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)
            interconv5 = slim.conv2d(
                pad(concat5), 512, 3, activation_fn=None, scope="interconv5")

            predict_flow5 = slim.conv2d(
                pad(interconv5),
                2,
                3,
                scope="predict_flow5",
                activation_fn=None)
            deconv4 = antipad(
                slim.conv2d_transpose(
                    concat5, 256, 4, stride=2, scope="deconv4"))
            upsample_flow5to4 = antipad(
                slim.conv2d_transpose(
                    predict_flow5,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow5to4",
                    activation_fn=None))
            concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)
            interconv4 = slim.conv2d(
                pad(concat4), 256, 3, activation_fn=None, scope="interconv4")

            predict_flow4 = slim.conv2d(
                pad(interconv4),
                2,
                3,
                scope="predict_flow4",
                activation_fn=None)
            deconv3 = antipad(
                slim.conv2d_transpose(
                    concat4, 128, 4, stride=2, scope="deconv3"))
            upsample_flow4to3 = antipad(
                slim.conv2d_transpose(
                    predict_flow4,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow4to3",
                    activation_fn=None))
            concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)
            interconv3 = slim.conv2d(
                pad(concat3), 128, 3, activation_fn=None, scope="interconv3")

            predict_flow3 = slim.conv2d(
                pad(interconv3),
                2,
                3,
                scope="predict_flow3",
                activation_fn=None)
            deconv2 = antipad(
                slim.conv2d_transpose(
                    concat3, 64, 4, stride=2, scope="deconv2"))
            upsample_flow3to2 = antipad(
                slim.conv2d_transpose(
                    predict_flow3,
                    2,
                    4,
                    stride=2,
                    scope="upsample_flow3to2",
                    activation_fn=None))
            concat2 = tf.concat([conv2, deconv2, upsample_flow3to2], axis=3)
            interconv2 = slim.conv2d(
                pad(concat2), 64, 3, activation_fn=None, scope="interconv2")

            predict_flow2 = slim.conv2d(
                pad(interconv2),
                2,
                3,
                scope="predict_flow2",
                activation_fn=None)
          # END: Refinement Network.

          flow = predict_flow2 * 0.05
          flow = tf.image.resize_bilinear(
              flow, tf.stack([height, width]), align_corners=True)

          return {
              "predict_flow6": predict_flow6,
              "predict_flow5": predict_flow5,
              "predict_flow4": predict_flow4,
              "predict_flow3": predict_flow3,
              "predict_flow2": predict_flow2,
              "flow": flow,
          }


class FlowNetCS:
  """Sequence of FlowNetC + FlowNetS."""

  def __init__(self):
    self.net_c = FlowNetC()
    self.net_s = FlowNetS()

  def run(self, inputs, trainable=True):
    """Runs model."""
    with tf.variable_scope("FlowNetCS"):
      # Forward pass through FlowNetC with weights frozen.
      net_c_predictions = self.net_c.run(inputs, trainable=False)

      # Perform flow warping (to move image B closer to image A based on flow
      # prediction).
      warped = flow_warp(inputs["input_b"], net_c_predictions["flow"])

      # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels).
      brightness_error = inputs["input_a"] - warped
      brightness_error = tf.square(brightness_error)
      brightness_error = tf.reduce_sum(brightness_error, keep_dims=True, axis=3)
      brightness_error = tf.sqrt(brightness_error)

      # Gather all inputs to FlowNetS
      inputs_to_s = {
          "input_a": inputs["input_a"],
          "input_b": inputs["input_b"],
          "warped": warped,
          "flow": net_c_predictions["flow"] * 0.05,
          "brightness_error": brightness_error,
      }

      return self.net_s.run(inputs_to_s, trainable=trainable)


class FlowNetCSS:
  """A sequence of FlowNetCS + FlowNetS."""

  def __init__(self):
    self.net_cs = FlowNetCS()
    self.net_s = FlowNetS()

  def run(self, inputs, trainable=True):
    """Runs model."""
    with tf.variable_scope("FlowNetCSS"):
      # Forward pass through FlowNetCS with weights frozen.
      net_cs_predictions = self.net_cs.run(inputs, trainable=False)

      # Perform flow warping (to move image B closer to image A based on flow
      # prediction).
      warped = flow_warp(inputs["input_b"], net_cs_predictions["flow"])

      # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels).
      brightness_error = inputs["input_a"] - warped
      brightness_error = tf.square(brightness_error)
      brightness_error = tf.reduce_sum(brightness_error, keep_dims=True, axis=3)
      brightness_error = tf.sqrt(brightness_error)

      # Gather all inputs to FlowNetS.
      inputs_to_s = {
          "input_a": inputs["input_a"],
          "input_b": inputs["input_b"],
          "warped": warped,
          "flow": net_cs_predictions["flow"] * 0.05,
          "brightness_error": brightness_error,
      }

      return self.net_s.run(inputs_to_s, trainable=trainable)


class FlowNet2:
  """Tensorflow implementation of FlowNet 2."""

  def __init__(self):
    self.net_css = FlowNetCSS()
    self.net_sd = FlowNetSD()

  def run(self, inputs, trainable=True):
    """Docstrings!!!"""
    _, height, width, _ = inputs["input_a"].shape.as_list()
    with tf.variable_scope("FlowNet2"):
      # Forward pass through FlowNetCSS and FlowNetSD with weights frozen.
      net_css_predictions = self.net_css.run(inputs, trainable=False)
      net_sd_predictions = self.net_sd.run(inputs, trainable=False)

      sd_flow_norm = channel_norm(net_sd_predictions["flow"])
      css_flow_norm = channel_norm(net_css_predictions["flow"])

      flow_warp_sd = flow_warp(inputs["input_b"], net_sd_predictions["flow"])
      img_diff_sd = inputs["input_a"] - flow_warp_sd
      img_diff_sd_norm = channel_norm(img_diff_sd)

      flow_warp_css = flow_warp(inputs["input_b"], net_css_predictions["flow"])
      img_diff_css = inputs["input_a"] - flow_warp_css
      img_diff_css_norm = channel_norm(img_diff_css)

      input_to_fusion = tf.concat([
          inputs["input_a"], net_sd_predictions["flow"],
          net_css_predictions["flow"], sd_flow_norm, css_flow_norm,
          img_diff_sd_norm, img_diff_css_norm
      ],
                                  axis=3)

      # Fusion Network.
      with slim.arg_scope(
          [slim.conv2d, slim.conv2d_transpose],
          # Only backprop this network if trainable.
          trainable=trainable,
          # He (aka MSRA) weight initialization.
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=leaky_relu,
          # We will do our own padding to match the original Caffe code.
          padding="VALID"):

        weights_regularizer = slim.l2_regularizer(WEIGHT_DECAY)
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=weights_regularizer):
          fuse_conv0 = slim.conv2d(
              pad(input_to_fusion), 64, 3, scope="fuse_conv0")
          fuse_conv1 = slim.conv2d(
              pad(fuse_conv0), 64, 3, stride=2, scope="fuse_conv1")
          fuse_conv1_1 = slim.conv2d(
              pad(fuse_conv1), 128, 3, scope="fuse_conv1_1")
          fuse_conv2 = slim.conv2d(
              pad(fuse_conv1_1), 128, 3, stride=2, scope="fuse_conv2")
          fuse_conv2_1 = slim.conv2d(
              pad(fuse_conv2), 128, 3, scope="fuse_conv2_1")

          predict_flow2 = slim.conv2d(
              pad(fuse_conv2_1),
              2,
              3,
              scope="predict_flow2",
              activation_fn=None)
          fuse_deconv1 = antipad(
              slim.conv2d_transpose(
                  fuse_conv2_1, 32, 4, stride=2, scope="fuse_deconv1"))
          fuse_upsample_flow2to1 = antipad(
              slim.conv2d_transpose(
                  predict_flow2,
                  2,
                  4,
                  stride=2,
                  scope="fuse_upsample_flow2to1",
                  activation_fn=None))
          concat1 = tf.concat(
              [fuse_conv1_1, fuse_deconv1, fuse_upsample_flow2to1], axis=3)
          fuse_interconv1 = slim.conv2d(
              pad(concat1), 32, 3, activation_fn=None, scope="fuse_interconv1")

          predict_flow1 = slim.conv2d(
              pad(fuse_interconv1),
              2,
              3,
              scope="predict_flow1",
              activation_fn=None)
          fuse_deconv0 = antipad(
              slim.conv2d_transpose(
                  concat1, 16, 4, stride=2, scope="fuse_deconv0"))
          fuse_upsample_flow1to0 = antipad(
              slim.conv2d_transpose(
                  predict_flow1,
                  2,
                  4,
                  stride=2,
                  scope="fuse_upsample_flow1to0",
                  activation_fn=None))
          concat0 = tf.concat(
              [fuse_conv0, fuse_deconv0, fuse_upsample_flow1to0], axis=3)
          fuse_interconv0 = slim.conv2d(
              pad(concat0), 16, 3, activation_fn=None, scope="fuse_interconv0")

          predict_flow0 = slim.conv2d(
              pad(fuse_interconv0),
              2,
              3,
              activation_fn=None,
              scope="predict_flow0")

          flow = tf.image.resize_bilinear(
              predict_flow0, tf.stack([height, width]), align_corners=True)
          return {
              "predict_flow0": predict_flow0,
              "flow": flow,
          }