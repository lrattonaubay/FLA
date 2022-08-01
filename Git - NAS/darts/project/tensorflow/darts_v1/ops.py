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
    self.batchnormalisation = tf.keras.layers.BatchNormalization(axis=-1,
                                                                 beta_initializer="random_uniform",
                                                                 gamma_initializer="random_uniform")
  
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
    
    elif x == ():
      out = ()
      print("décommenter 4")
    else:
      print("décommenter 3")
      """
      # x is a list
      out = ()
      for i in range(len(x)):
        if isinstance(len(x[i]),int):
          # we want to know if x[i] is a tensor list
          if len(x[i]) == 0 :
            # x[i] contains no element
            print("x[i_pool_bn] est vide")
          elif len(x[i]) == 1 :
            # x[i] contains only one tensor
            entree = x[i][0]
            x_i = self.pooling(entree)
            out_i = self.batchnormalisation(x_i)
            out = out + (out_i,)
          else:
            # x[i] is a list containing more than one tensor
            for j in range(len(x[i])):
              entree = x[i][j]
              if entree.get_shape()[1] >= 3 and entree.get_shape()[2] >= 3:
                # We check that the dimensions of the tensor respect the conditions necessary to apply the layer.
                x_i_j = self.pooling(entree)
                out_i_j = self.batchnormalisation(x_i_j)
                out = out + (out_i_j,)
              else:
                out_i_j = tf.constant(1,shape=(3, 3, 3, 4))
                out = out + (out_i_j,)            
        else:
          # x[i] is a Tensor
          x_i = self.pooling(x[i])
          out_i = self.batchnormalisation(x_i)
          out = out + (out_i,)
    """
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
    self.relu = tf.keras.layers.ELU()
    self.conv2d = tf.keras.layers.Conv2D(filters=C_out,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        padding=padding, 
                                        use_bias=False)
    self.batch_normalisation = tf.keras.layers.BatchNormalization(axis=-1,
                                                                  beta_initializer="random_uniform",
                                                                  gamma_initializer="random_uniform")
  
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
    
    else:
      print("décommenter 2")
      """
      # x is a list
      out = ()
      for i in range(len(x)):
        if isinstance(len(x[i]),int):
          # we want to know if x[i] is a tensor list
          if len(x[i]) == 0 :
            # x[i] contains no element
            print("x[i_std_conv] est vide")
          elif len(x[i]) == 1 :
            # x[i] contains only one tensor
            entree = x[i][0]
            
            #print("entree.get_shape() = {}".format(entree.get_shape()))
            x_i = self.relu(entree)
            out_conv2d_i = self.conv2d(x_i)
            
            out_i = self.batch_normalisation(x_i)
            
            out = out + (out_i,)
          else:
            # x[i] is a list containing more than one tensor
            for j in range(len(x[i])):
              entree = x[i][j]
              x_i_j = self.relu(entree)
              out_conv2d_i_j = self.conv2d(x_i_j)
              out_i_j = self.batch_normalisation(out_conv2d_i_j)
              out = out + (out_i_j,)
            
        else:
          # x[i] is a Tensor

          x_i = self.relu(x[i])
          out_conv2d_i = self.conv2d(x_i)
          out_i = self.batch_normalisation(out_conv2d_i)
          out = out + (out_i,)
    """
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
    self.net.add(tf.keras.layers.ELU())
    self.net.add(tf.keras.layers.Conv2D(C_in, kernel_size, strides=stride, dilation_rate=dilation, padding="valid", groups=C_in, data_format="channels_last" ) )
    self.net.add(tf.keras.layers.Conv2D(C_out, kernel_size, strides=(1, 1), padding='valid', data_format="channels_last"))
    self.net.add(tf.keras.layers.BatchNormalization(axis=-1,
                                                    beta_initializer="random_uniform",
                                                    gamma_initializer="random_uniform") )

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
    self.batch_normalisation = tf.keras.layers.BatchNormalization(axis=-1,
                                                                  beta_initializer="random_uniform",
                                                                  gamma_initializer="random_uniform")

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
      x2 = tf.keras.layers.ELU()(x)
      
      out_1 = self.conv2d_1(x2)
      out_2 = self.conv2d_2(x2[:,:,1:,1:])
      out_3 = self.concatenation([out_1,out_2])
      
      out = self.batch_normalisation(out_3)
    
    else:
      print("décommenter 1")
      """
      # x is a list
      out = ()
      for i in range(len(x)):
        if isinstance(len(x[i]),int):
          # we want to know if x[i] is a tensor list
          if len(x[i]) == 0 :
            # x[i] contains no element
            out = ()
          elif len(x[i]) == 1 :
            # x[i] contains only one tensor
            entree = x[i][0]
            #print("entree.get_shape() = {}".format(entree.get_shape()))
            x2_i = tf.keras.layers.ELU()(entree)
            out_1_i = self.conv2d_1(x2_i)
            out_2_i = self.conv2d_2(x2_i[:,:,1:,1:])
            out_3_i = self.concatenation([out_1_i,out_2_i])
            out_i = self.batch_normalisation(out_3_i)
            out = out + (out_i,)
          else:
            # x[i] is a list containing more than one tensor
            for j in range(len(x[i])):
              entree = x[i][j]
              x2_i_j = tf.keras.layers.ELU()(entree)
              out_1_i_j = self.conv2d_1(x2_i_j)
              out_2_i_j = self.conv2d_2(x2_i_j[:,:,1:,1:])
              out_3_i_j = self.concatenation([out_1_i_j,out_2_i_j])
              out_i_j = self.batch_normalisation(out_3_i_j)
              out = out + (out_i_j,)
            
        else:
          # x[i] is a Tensor
          x2_i = tf.keras.layers.ELU()(x[i])
          out_1_i = self.conv2d_1(x2_i)
          out_2_i = self.conv2d_2(x2_i[:,:,1:,1:])
          out_3_i = self.concatenation([out_1_i,out_2_i])
          out_i = self.batch_normalisation(out_3_i)
          out = out + (out_i,)
    """
    return out