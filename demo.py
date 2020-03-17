import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

import flowlib
import models


def load_image(path: str) -> np.ndarray:
  img = cv2.imread(path, cv2.IMREAD_COLOR)

  # OpenCV stores images as BGR, but the model expects RGB.
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Convert to float and add batch dimension.
  return np.expand_dims(img.astype(np.float32) / 255., axis=0)


if __name__ == "__main__":
  # Graph construction.
  image1_placeholder = tf.placeholder(tf.float32, [1, 384, 512, 3])
  image2_placeholder = tf.placeholder(tf.float32, [1, 384, 512, 3])
  flownet2 = models.FlowNet2()
  inputs = {"input_a": image1_placeholder, "input_b": image2_placeholder}
  predicted_flow = flownet2.run(inputs)["flow"]

  # Load inputs.
  image1 = load_image("example/0img0.ppm")
  image2 = load_image("example/0img1.ppm")

  ckpt_file = "checkpoints/FlowNet2/flownet-2.ckpt-0"
  saver = tf.train.Saver()

  with tf.Session() as sess:
    saver.restore(sess, ckpt_file)
    feed_dict = {image1_placeholder: image1, image2_placeholder: image2}
    predicted_flow_np = sess.run(predicted_flow, feed_dict=feed_dict)

  # Show visualization.
  flow_image = flowlib.flow_to_image(predicted_flow_np[0])
  plt.imshow(flow_image)
  plt.show()
