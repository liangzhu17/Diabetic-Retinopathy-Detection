
# grad cam implementation

def grad_cam(model, image, class_index, layer_name):
  """
    Args:
       model: model
       image: image input
       class_index: label
       layer_name: last convolution layer name
    """
  
  grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

  # obtain class output
  class_output = model.output[:,class_index]

  # obtain the output and loss of the layer which we want
  with tf.GradientTape() as tape:
    convolution_output, predictions = grad_model(image)
    loss = predictions[:, class_index]
  output = convolution_output[0]

  # calculate the gradients
  grads = tape.gradient(loss, convolution_output)[0]
  
  #get average weights
  weights = np.mean(grads, axis=(0, 1))

  # class activation mapping
  cam = np.ones(output.shape[0:2], dtype=np.float32)
  for i, w in enumerate(weights):
    cam += w * output[:, :, i]

  # express the gradient weight in RGB
  cam = cv2.resize(cam.numpy(), (256, 256))
  cam = np.maximum(cam, 0)
  heatmap = (cam - cam.min()) / (cam.max() - cam.min())
  cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

  #plot grad cam visualization
  fig=plt.figure(figsize=(15, 5))
  fig.add_subplot(1, 2, 1)
  plt.title("Original Image")
  plt.imshow(img)
  fig.add_subplot(1, 2, 2)
  plt.title("Deep Visualization")
  plt.imshow(cam)
  plt.show()


# visualize the test dataset
test_ds_unbatch = test_ds.unbatch()
for i, l in test_ds_unbatch:
  img = i
  image = tf.expand_dims(i, 0)   # get input image
  label = np.argmax(l,axis=0)    # get true label of the input image
  pred = model.predict(image)    # get predict label of the input image

  grad_cam(model, image, label, "conv_7") # get gradcam visualization of each image in test dataset
