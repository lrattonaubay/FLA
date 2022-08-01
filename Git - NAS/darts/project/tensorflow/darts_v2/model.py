from collections import OrderedDict
import ops
from choices import LayerChoice, InputChoice
import tensorflow as tf


class Node(tf.keras.layers.Layer):

    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        """
        """
        super().__init__()
        self.ops = []
        choice_keys = []

        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                LayerChoice(OrderedDict([
                                        ("maxpool", ops.PoolBN('max', channels, 3, stride, "same", affine=False)),
                                        ("avgpool", ops.PoolBN('avg', channels, 3, stride, "same", affine=False)),
                                        ("skipconnect",ops.Identity() if stride == 1 else ops.FactorizedReduce(channels,channels,affine=False)),
                                        #("sepconv3x3", ops.SepConv(channels, channels, 3, stride, 1, affine=False)),
                                        #("sepconv5x5", ops.SepConv(channels, channels, 5, stride, 2, affine=False)),
                                        #("dilconv3x3", ops.DilConv(channels, channels, 3, stride, 2, 2, affine=False)),
                                        #("dilconv5x5", ops.DilConv(channels, channels, 5, stride, 4, 2, affine=False))
                                        ]), label=choice_keys[-1]))
        
        self.drop_path = ops.DropPath()
        self.input_switch = InputChoice(n_candidates=len(choice_keys),n_chosen=2,label="{}_switch".format(node_id))
        
    def call(self, prev_nodes):

        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out] 
        return self.input_switch(out)


class Cell(tf.keras.layers.Layer):

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, "valid", affine=False)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, "valid", affine=False)

        # generate dag

        self.mutable_ops = []

        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth), 
                                    depth, channels, 2 if reduction else 0))

    def call(self, s0, s1):
    # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = tf.concat(tensors[2:], axis=-1)
        return output

class CNN(tf.keras.layers.Layer):
  """
   Description
   ---------------
   This class allows the creation of a CNN model.
  """
  def __init__(self, input_size, in_channels, channels, n_classes, n_layers, n_nodes=4,
               stem_multiplier=3, auxiliary=False):
    """
    Description
    ---------------
    This function creates an instance of the CNN class.
    Input(s)
    ---------------
    input_size: int
    in_channels: int
    channels: int
    n_classes: int
    n_layers: int
    n_nodes: int
    stem_multiplier: int
    auxiliary: bool
    Output(s)
    ---------------
    no output
    """
    super().__init__()
    self.in_channels = in_channels
    self.channels = channels
    self.n_classes = n_classes
    self.n_layers = n_layers
    self.aux_pos = 2 * n_layers // 3 if auxiliary else -1
    c_cur = stem_multiplier * self.channels
    self.conv2d = tf.keras.layers.Conv2D(filters=c_cur,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         kernel_size=1,
                                         use_bias=False )
    self.batch_normalisation = tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform", axis=-1)
    # for the first cell, stem is used for both s0 and s1
    # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
    channels_pp, channels_p, c_cur = c_cur, c_cur, channels
    self.cells = []
    reduction_p, reduction = False, False
    for i in range(n_layers):
      reduction_p, reduction = reduction, False
      # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
      if i in [n_layers // 3, 2 * n_layers // 3]:
        c_cur *= 2
        reduction = True
      cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
      self.cells.append(cell)
      c_cur_out = c_cur * n_nodes
      channels_pp, channels_p = channels_p, c_cur_out

    self.gap = tf.keras.layers.GlobalAveragePooling2D()
    self.linear = tf.keras.layers.Dense(n_classes)
  
  def GetLogits(self, x):
    """
	Description
    ---------------
    This function creates a list of tensor that is the result
	of the self.linear operation applied on x.
    Input(s)
    ---------------
    x : Tensor or list of tensor
    Output(s)
    ---------------
    out : list of Tensor
    """
    out = []
    if isinstance(x,tf.Tensor):
        # x is a Tensor
        out = self.linear(x)
    elif x == ():
         out = []
    else:
        # x is a list
        for i in range(len(x)):
            if isinstance(len(x[i]),int):
                # we want to know if x[i] is a tensor list
                if len(x[i]) == 0 :
                    # x[i] contains no element
                    out = out
                elif len(x[i]) == 1 :
                    # x[i] contains only one tensor
                    entree = x[i][0]
                    out_i = self.linear(entree)
                    out.append(out_i)
                else:
                    # x[i] is a list containing more than one tensor
                    for j in range(len(x[i])):
                        entree = x[i][j]

                        if entree.get_shape()[1] >= 3 and \
                           entree.get_shape()[2] >= 3:
                            # We check that the dimensions of the tensor 
                            # respect the conditions necessary to apply the layer.
                            out_i_j = self.linear(entree)
                            out.append(out_i_j)
                        else:
                            out_i_j = tf.constant(1,shape=(3, 3, 3, 4))
                            out.append(out_i_j)
            else:
                # x[i] is a Tensor
                out_i = self.flatten(x[i])
                out.append(out_i)
    
    return out
  
  def ApplyLayer(self, fonction, x):
    """
	Description
    ---------------
    This function creates a tuple of tensor that is the result
	of the self.linear operation applied on x.
    Input(s)
    ---------------
    x : Tensor or list of tensor
    Output(s)
    ---------------
    out : tuple of Tensor
    """
    if isinstance(x,tf.Tensor):
        # x is a Tensor
        out = fonction(x)
    elif x == ():
         out = ()
    else:
        # x is a list
        out = ()
        x = tf.convert_to_tensor(x)

        for i in range(len(x)):
            if isinstance(len(x[i]),int):
                # we want to know if x[i] is a tensor list
                if len(x[i]) == 0 :
                    # x[i] contains no element
                    out = out + ()
                elif len(x[i]) == 1 :
                    # x[i] contains only one tensor
                    entree = x[i][0]
                    out_i = fonction(entree)
                    out = out + (out_i,)
                else:
                    # x[i] is a list containing more than one tensor
                    for j in range(len(x[i])):
                        entree = x[i][j]

                        if entree.get_shape()[1] >= 3 and \
                           entree.get_shape()[2] >= 3:
                            # We check that the dimensions of the tensor 
                            # respect the conditions necessary to apply the layer.
                            out_i_j = fonction(entree)
                            out = out + (out_i_j,)
                        else:
                            out_i_j = tf.constant(1,shape=(3, 3, 3, 4))
                            out = out + (out_i_j,)            
            else:
                # x[i] is a Tensor
                out_i = fonction(x[i])
                out = out + (out_i,)
    
    return out



  
  def call(self, x):
    """
    Description
    ---------------
    This function applies the instance of the CNN class to the data.
    Input(s)
    ---------------
    x: tf.Tensor
    Output(s)
    ---------------
    logits : tf.Tensor
    """
    out1 = self.conv2d(x)
    s0 = self.batch_normalisation(out1)
    s1 = self.batch_normalisation(out1)
    aux_logits = None
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
      if i == self.aux_pos and self.training:
        aux_logits = self.aux_head(s1)

    out = self.ApplyLayer(self.gap, s1)
    self.flatten = tf.keras.layers.Flatten()
    out2 = self.ApplyLayer(self.flatten, out)
    logits = self.GetLogits(out2)
    if aux_logits is not None:
      return logits, aux_logits
    return logits

  def drop_path_prob(self, p):
    """
    Description
    ---------------
    This function drops some paths between neurons of the CNN model
    with a probability p.
    Input(s)
    ---------------
    p: float
    Output(s)
    ---------------
    no output
    """
    for module in self.modules():
      if isinstance(module, ops.DropPath):
        module.p = p
