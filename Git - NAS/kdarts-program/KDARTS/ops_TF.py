import tensorflow as tf

def PoolBN(pool_type, C, kernel_size, stride, padding, x):
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
    padding: int
    x: previous layer

    Output(s)
    ---------------
    out ; Tensor
    """
    # self.architecture = tf.keras.models.Sequential()
    if pool_type == "max":
      # self.architecture.add(tf.keras.layers.MaxPooling2D(kernel_size, stride, padding))
      pool = tf.keras.layers.MaxPooling2D(kernel_size, stride, padding)
    elif pool_type == "avg":
      # self.architecture.add(tf.keras.layers.AveragePooling2D(kernel_size, stride, padding))
      pool = tf.keras.layers.AveragePooling2D(kernel_size, stride, padding)
    else:
      raise ValueError()
    BN = tf.keras.layers.BatchNormalization()

    # print('pooling',x)
    out = pool(x)
    out = BN(out)
    return out
	

def FactorizedReduce(C_in, C_out, x, affine=True):
    """
    Description
    ---------------
    This function initializes an instance of the FactorizedReduce class.
    Input(s)
    ---------------
    C_in: int
    C_out: int
    x: previous layer
    Output(s)
    ---------------
    No output
    """
    relu = tf.keras.layers.ReLU()
    conv2d_1 = tf.keras.layers.Conv2D(filters=C_out // 2 + C_out % 2,
                                      strides=2,
                                      padding='same' ,
                                      data_format="channels_last",
                                      kernel_size=1,
                                      use_bias=False)
    conv2d_2 = tf.keras.layers.Conv2D(filters=C_out // 2,
                                      strides=2,
                                      padding='same' ,
                                      data_format="channels_last",
                                      kernel_size=1,
                                      use_bias=False)
    concatenation = tf.keras.layers.Concatenate()
    batch_normalisation = tf.keras.layers.BatchNormalization()
    
    x2 = tf.keras.layers.ReLU()(x)
    out_1 = conv2d_1(x2)
    out_2 = conv2d_2(x2[:,1:,1:,:])
    out_conc = concatenation([out_1,out_2])
    out = batch_normalisation(out_conc)
    return out


def Skipconnect(x, affine=True):
    """
    Description
    ---------------
    This function initializes an instance of the FactorizedReduce class.
    Input(s)
    ---------------
    x: previous layer
    Output(s)
    ---------------
    No output
    """
    return x


def DilConv(C_in, C_out, kernel_size, stride, padding, dilation, x):
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
    x: previous layer
    Output(s)
    ---------------
    No output
    """

    x2 = tf.keras.layers.ReLU()(x)
    conv_1 = tf.keras.layers.Conv2D(C_in, kernel_size, strides=stride, padding="same",  data_format="channels_last" )(x2)
    conv_2 = tf.keras.layers.Conv2D(C_out, kernel_size, strides=(1, 1), dilation_rate=dilation, padding=padding, data_format="channels_last")(conv_1)
    BN = tf.keras.layers.BatchNormalization()(conv_2)
    return BN

def SepConv(C_in, C_out, kernel_size, stride, padding, x):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """
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
    x: previous layer
    Output(s)
    ---------------
    No output
    """

    x2 = DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, x=x)
    out = DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, x=x2)
    return out


"""
Description
---------------
This class is used to do an average convolution.
It is composed of a relu followed by the convolution.
Then a batchnormalisation is applied.
The model used is a sequential keras model.
"""

def StdConv(C_in, C_out, kernel_size, stride, padding, x):
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
    x: previous layer

    Output(s)
    ---------------
    No output
    """

    relu = tf.keras.layers.ReLU()
    conv2d = tf.keras.layers.Conv2D(filters=C_out,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding=padding, 
                                    use_bias=False )
    BN = tf.keras.layers.BatchNormalization(axis=-1)


    out_relu = relu(x)
    out_conv2d = conv2d(out_relu)
    out_batch_normalisation = BN(out_conv2d)
    out = out_batch_normalisation

    return out

