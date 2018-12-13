'''
USE:

import numpy as np
import cnn_extract

XY = np.random.randint(255, size=(300, 7320))
Y = np.random.choice((0,1),size=(300,1))
XY = np.hstack((Y,XY))

feat_dict = cnn_extract.extract(XY,50,6,peek=False)

#Now feat_dict contains 4 feature arrays of your dataset
#lets check the features right before the dense layer of the cnn
print(feat_dict['features'])
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO) #This way we can see the training information

def cnn_model_fn(features, labels, mode):
  """
  A CNN model for a the extract method
  """
  
  #Input preprocessing layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # spectographs are 40x183 pixels, and have one color channel
  input_layer = tf.reshape(features["x"],[-1,40,183,1]) #-1 means adjust batch size so that feature data fills the dimension of (40,183,1) before the starting the next batch element
  
  #Module 1: Extraction
  # Computes 32 new features using a 7x11 filter with ReLU activation. Filter dimensions preserve ratios of convolution 7/40~5/28, 11/(183/3)~5/28
  # Padding is added to preserve width and height during convolution.
  # The (1,1) stride makes each new feature have same dimensions as input spectrograph
  # Input Tensor Shape: [batch_size, 40, 183, 1]
  # Output Tensor Shape: [batch_size, 40, 183, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[7,11],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu)
  # First max pooling layer with a 3x5 filter and stride of 2. Filter dimensions preserve ratios of shrinking filters 3/7~2/5, 5/11~2/5
  # The stride cuts the X's image dimensions by ceil(2)
  # Retains the newest 32 features
  # Input Tensor Shape: [batch_size, 40, 183, 32]
  # Output Tensor Shape: [batch_size, 20, 92, 32]
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1,
      pool_size=[3,5],
      strides=(2,2),
      padding='same',
  )
  
  #Module 2: Extraction
  # Computes 32 new features, totaling 32+32=64 using a 7x11 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 20, 92, 32]
  # Output Tensor Shape: [batch_size, 20, 92, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[7,11],
      strides=(1, 1),
      padding="same",
      activation=tf.nn.relu
  )
  # Second max pooling layer with a 3x5 filter and stride of 2
  # The stride cuts the X's image dimensions by ceil(2)
  # Retains the total 64 features
  # Input Tensor Shape: [batch_size, 20, 92, 64]
  # Output Tensor Shape: [batch_size, 10, 46, 64]
  pool2 = tf.layers.max_pooling2d(
      inputs=conv2,
      pool_size=[3,5],
      strides=(2,2),
      padding='same'
  )
  
  #Module 3: Prediction
  # Flatten tensor into a batch of vectors, just like our input
  # Input Tensor Shape: [batch_size, 10, 46, 64]
  # Output Tensor Shape: [batch_size, 10 * 46 * 64 = 29,440]
  pool2_flattened = tf.reshape(
      pool2,
      [-1, 10 * 46 * 64]
  )
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 10 * 46 * 64 = 29,440]
  # Output Tensor Shape: [batch_size, 4096 = 2^12]
  dense = tf.layers.dense(
      inputs=pool2_flattened,
      units=4096,
      activation=tf.nn.relu)
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense,
      rate=0.4,
      training= (mode==tf.estimator.ModeKeys.TRAIN) )
  # Input Tensor Shape: [batch_size, 4096 = 2^12]
  # Output Tensor Shape: [batch_size, 2 = 2^1]
  logits = tf.layers.dense(inputs=dropout, units=2)
  
  #Modes
  #Based on the mode, return a different value
  #Modes include: Train, Test, Predict, Eval
  
  #  Predict Mode
  predictions = {
      #Generate the logits predictions as a dictionary
      #  One key for the flattened convolution features of each input image
      "features_pool2_flattened": pool2_flattened,
      #  One key for the dense features of each input image
      "features_dense": dense,
      #  One key for the predicted classes from logits
      "classes": tf.argmax(input=logits,axis=1),
      #  One key of the overal logits vector shape
      "probabilities": tf.nn.softmax(logits,name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  #  Train & Eval Mode
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  #  Train
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )
  
  #  Eval
  eval_metric_ops = {
      #Generate the evaluation metrics as a dictionary
      "accuracy": tf.metrics.accuracy(
          labels=labels,
          predictions=predictions["classes"]
      )
  }
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      eval_metric_ops=eval_metric_ops
  )

def extract(XY, batch_arg, steps_arg, Xtest=None, peek=False):
  '''
  Inputs: (XY, batch_arg, steps_arg, Xtest=None, peek=False)
   XY        - entire training dataset
   batch_arg - size of batches
   steps_arg - number of batches to train over
   Xtest     - dataset for which to extract features
               if ommited features will be extracted for the XY dataset
   peek      - set to True to view probability vectors while training
   
   Outputs: {'features':flat_r,'features_dense':dense_r,'predictions':pred_r,'probabilities': prob_r}
    dictionary- of feature vectors for each entry in your XY or X dataset
     'features' - has dimension (M,29440)
     'features_dense' - has dimension (M,4096)
     'predictions' - has dimension (M,)
     'probabilities' - has dimension (M,2)
    this method also graphs the first image of the XY or X dataset
  '''
  Y = np.asarray(XY[:,0],dtype=np.int32) #(M,)
  X = np.asarray(XY[:,1:],dtype=np.float32) #(M,7320)
  
  print(f' Type XY{type(XY)}\n Type Y{type(Y)}\n Type X{type(X)}')
  print(f' Y shape {Y.shape}\n X shape {X.shape}')
  
  #log appropriately
  #https://stackoverflow.com/questions/46013115/how-to-change-global-step-in-tensorflows-skcompat
  config = tf.estimator.RunConfig(
    save_summary_steps=np.maximum(1,steps_arg//10),
    log_step_count_steps=np.maximum(1,steps_arg//10) #this is where we display log outputs. should be a factor of training "steps" below
  )

  #create correct estimator using the custom "cnn_model_fn" defined above
  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="/tmp/tf_cnn_spectograph_model",
    config=config
  )
  
  #log progress while training
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log,
    every_n_iter=np.maximum(1,batch_arg//2) #Only affect the long log output. Does not affect how frequently we see "steps" in output. If not divisible by number of steps, it will take precedence. IE log_iter=5, #steps=11 -> Training will last for 16steps
  )
  
  #prepare training input with shuffling, batching, etc
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X},
    y=Y,
    batch_size=batch_arg,
    num_epochs=None,
    shuffle=True
  )
  
  ##train the model by calling "estimator.train"
  mnist_classifier.train(
   input_fn=train_input_fn,
   steps=steps_arg, #Batch size = 50, total data = 300, so there are 6 steps in one epoch, we'll do 4 epochs or 32 steps over a total of 1,200 images
   hooks= ([logging_hook] if peek else None)
   )
  
  #feature_input_fn = tf.estimator.inputs.numpy_input_fn(
  #    x={"x":X},
  #    num_epochs=1,
  #    shuffle=False)
  
  if (Xtest != None):
   images = Xtest
  else:
   images = X

  feature_input_fn = tf.estimator.inputs.numpy_input_fn(
   x={"x":images},
   num_epochs=1,
   shuffle=False
   )

  #a = mnist_classifier.predict(input_fn=feature_input_fn) #Creates something called a generator or iterator object in python. The only way to access its output is by iterating through with "next()" or a "for" loop
  #print(next(a)['features_pool2_flattened'])
  
  M = images.shape[0]
  flat_r = np.zeros((M,29440))
  dense_r = np.zeros((M,4096))
  pred_r = np.zeros(M)
  prob_r = np.zeros((M,2))
  
  print('first image')
  plt.imshow(images[0].reshape(40, 183), cmap=plt.cm.binary) #Print the original image we are predicting
  plt.show() #Show the plot
  
  #count = 0
  for index,gen in zip(range(M),mnist_classifier.predict(input_fn=feature_input_fn)):
   flat_r[index,:]=gen['features_pool2_flattened']
   dense_r[index,:]=gen['features_dense']
   pred_r[index]=gen['classes']
   prob_r[index,:]=gen['probabilities']
   #count = count + 1
  #print(count)
  return {'features':flat_r,'features_dense':dense_r,'predictions':pred_r,'probabilities': prob_r}
