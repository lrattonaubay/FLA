import tensorflow as tf

class DropPath(tf.keras.layers.Layer):
  def __init__(self, p=0):
    """
    Description
    ---------------
    Creates an instance of the class DropPath.

    Input(s)
    ---------------
    p: float (0 if you don't want, else, between 0 and 1)

    Output(s)
    ---------------
    No output
    """
    super().__init__()
    self.p=p
  
  def call(self, x):
    """
    Description:
    ---------------
    Similar to the forward function in tensorflow, this function is used to train the AI.

    Input(s)
    ---------------
    x : tf.Tensor

    Output(s)
    ---------------
    output: tf.Tensor
    """
    if self._trainable and self.p>0:
      keep_prob = 1. - self.p
      shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
      mask = tf.keras.backend.random_bernoulli(shape, keep_prob, dtype = tf.float32)
      return x/keep_prob*mask
    return x

class PoolBN(tf.keras.layers.Layer):
  """
  Description
  ---------------
  This class is used to handle both the max pooling layer and the average pooling layer.
  In both cases, the pooling is followed by a layer of batch normalization.
  """
  def __init__(self, pool_type, C, kernel_size, stride, padding, affine=False):
    """
    Description
    ---------------
    Creates an instance of the class PoolBN

    Input(s)
    ---------------
    pool_type: string (either "max" or "avg")
    C: int, axis on which to perform batch normalization
    kernel_size: int or enumerable of ints
    stride: int or enumerable of ints
    padding: string
    affine: bool

    Output(s)
    ---------------
    No output
    """
    super().__init__()
    if pool_type == "max":
      self.pooling = tf.keras.layers.MaxPooling2D(kernel_size, stride, padding)
    elif pool_type == "avg":
      self.pooling = tf.keras.layers.AveragePooling2D(kernel_size, stride, padding)
    else:
      raise ValueError()
    self.batchnormalisation = tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform")
  
  def call(self, x):
    """
    Description:
    ---------------
    Similar to the forward function in tensorflow, this function is used to train the AI.

    Input(s)
    ---------------
    x : tf.Tensor

    Output(s)
    ---------------
    output: tf.Tensor
    """

    if isinstance(x,tf.Tensor):
      # x is a Tensor
      out1 = self.pooling(x)
      out = self.batchnormalisation(out1)
    
    return out
    
class StdConv(tf.keras.layers.Layer):
  """
  Description
  ---------------
  This class is used to do an average convolution.
  It is composed of a relu followed by the convolution.
  Then a batchnormalisation is applied.
  The model used is a sequential keras model.
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    """
    Description
    ---------------
    Creates an instance of the class StdConv

    Input(s)
    ---------------
    C_in: int
    C_out: int
    kernel_size: int
    stride: int
    padding: string ("same" ou "valid")
    affine: bool

    Output(s)
    ---------------
    No output
    """
    super().__init__()
    #self.relu = tf.keras.layers.ReLU()
    #self.relu = tf.keras.layers.LeakyReLU()
    self.relu = tf.keras.layers.ELU()
    self.conv2d = tf.keras.layers.Conv2D(filters=C_out,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        padding=padding, 
                                        bias_initializer='ones',
                                        use_bias=False)
    self.batch_normalisation = tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform", axis=-1)
  
  def call(self, x):
    """
    Description:
    ---------------
    Similar to the forward function in tensorflow, this function is used to train the AI.

    Input(s)
    ---------------
    x : tf.Tensor

    Output(s)
    ---------------
    output: tf.Tensor
    """
    if isinstance(x,tf.Tensor):
      # x is a Tensor
      out_relu = self.relu(x)
      out_conv2d = self.conv2d(out_relu)
      out_batch_normalisation = self.batch_normalisation(out_conv2d)
      out = out_batch_normalisation
  
    return out

class DilConv(tf.keras.layers.Layer):
  """
  This class is the dilated depthwise separable convolution class.
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    """
    Description
    ---------------
    This function initializes an instance of the DilConv class.
    Input(s)
    ---------------
    C_in: int
    C_out: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    affine: boolean
    Output(s)
    ---------------
    No output
    """
    super().__init__()

    self.net = tf.keras.Sequential()
    #self.net.add(tf.keras.layers.ReLU())
    #self.net.add(tf.keras.layers.LeakyReLU())
    self.net.add(tf.keras.layers.ELU())
    self.net.add(tf.keras.layers.Conv2D(C_in, kernel_size, strides=stride, padding="same", groups=C_in, data_format="channels_last" ))
    self.net.add(tf.keras.layers.Conv2D(C_out, kernel_size, strides=(1, 1), dilation_rate=dilation, padding="same", data_format="channels_last"))

    self.net.add(tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform"))

  def call(self, x):
    """
    Description
    ---------------
    This function applies the instance of the DilConv class to the data.
    Input(s)
    ---------------
    x: tf.Tensor
    Output(s)
    ---------------
    output: tf.Tensor
    """

    return self.net(x)



class SepConv(tf.keras.layers.Layer):
  """
  Depthwise separable conv.
  DilConv(dilation=1) * 2.
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    """
    Description
    ---------------
    This function initializes an instance of the SepConv class.
    Input(s)
    ---------------
    C_in: int
    C_out: int
    kernel_size: int
    stride: int
    padding: int
    affine: boolean
    Output(s)
    ---------------
    No output
    """
    super().__init__()
    self.net = tf.keras.Sequential()
    self.net.add(DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1,affine=affine))
    self.net.add(DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine))

  def call(self, x):
    """
    Description
    ---------------
    This function applies the instance of the SepConv class to the data.
    Input(s)
    ---------------
    x: tf.Tensor
    Output(s)
    ---------------
    output: tf.Tensor
    """
    out = self.net(x)
    return self.net(x)

class FactorizedReduce(tf.keras.layers.Layer):

  """

  Reduce feature map size by factorized pointwise (stride=2).

  """

  def __init__(self, C_in, C_out, affine=True):
    """
    Description
    ---------------
    This function initializes an instance of the FactorizedReduce class.
    Input(s)
    ---------------
    C_in: int
    C_out: int
    Output(s)
    ---------------
    No output
    """
    super().__init__()
    #self.relu = tf.keras.layers.ReLU()
    #self.relu = tf.keras.layers.LeakyReLU()
    self.relu = tf.keras.layers.ELU()
    self.conv2d_1 = tf.keras.layers.Conv2D(filters=C_out // 2,
                                      strides=2,
                                      padding='same' ,
                                      data_format="channels_last",
                                      kernel_size=C_in,
                                      use_bias=False)
    self.conv2d_2 = tf.keras.layers.Conv2D(filters=C_out // 2,
                                      strides=2,
                                      padding='same' ,
                                      data_format="channels_last",
                                      kernel_size=C_in,
                                      use_bias=False)
    self.concatenation = tf.keras.layers.Concatenate(axis=-1)
    self.batch_normalisation = tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform", axis=-1)

  def call(self, x):
    """
    Description
    ---------------
    This function applies the instance of the FactorizedReduce class to the data.
    Input(s)
    ---------------
    x: tf.Tensor
    Output(s)
    ---------------
    output: tf.Tensor
    """
    if isinstance(x,tf.Tensor):
      # x is a Tensor
      #x2 = tf.keras.layers.relu()(x)
      x2 = tf.keras.layers.LeakyReLU()(x)

      out_1 = self.conv2d_1(x2)
      out_2 = self.conv2d_2(x2[:,:,1:,1:])
      out_3 = self.concatenation([out_1,out_2])
      
      out = self.batch_normalisation(out_3)
    
    return out


class Identity(tf.keras.layers.Layer):
  """
  Description
  ---------------
  This class is used to handle both the max pooling layer and the average pooling layer.
  In both cases, the pooling is followed by a layer of batch normalization.
  """
  def __init__(self):
    super().__init__()
    """
    Description
    ---------------
    Creates an instance of the class PoolBN

    Input(s)
    ---------------
    pool_type: string (either "max" or "avg")
    C: int, axis on which to perform batch normalization
    kernel_size: int or enumerable of ints
    stride: int or enumerable of ints
    padding: string
    affine: bool

    Output(s)
    ---------------
    No output
    """
  
  def call(self, x):
    """
    Description:
    ---------------
    Similar to the forward function in tensorflow, this function is used to train the AI.

    Input(s)
    ---------------
    x : tf.Tensor

    Output(s)
    ---------------
    output: tf.Tensor
    """
    
    return tf.identity(x)