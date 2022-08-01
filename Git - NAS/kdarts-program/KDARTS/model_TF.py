import tensorflow as tf
from ops_TF import PoolBN, FactorizedReduce, Skipconnect, DilConv, SepConv, StdConv

def Node_func(element, channels, reducing, input):
  outs = []
  inputs = input
  for key in element:
    stride = 2 if reducing and int(key)<2 else 1
    dict_element = {
      "maxpool":PoolBN("max", channels, 3, stride, "same", inputs[int(key)]),
      "avgpool":PoolBN("avg", channels, 3, stride, "same", inputs[int(key)]),
      "skipconnect":Skipconnect(inputs[int(key)]) if stride == 1 else FactorizedReduce(channels, channels, inputs[int(key)]),
      "sepconv3x3":SepConv(channels, channels, 3, stride, "same", inputs[int(key)]),
      "sepconv5x5":SepConv(channels, channels, 5, stride, "same", inputs[int(key)]),
      "dilconv3x3":DilConv(channels, channels, 3, stride, "same", 2, inputs[int(key)]),
      "dilconv5x5":DilConv(channels, channels, 5, stride, "same", 2, inputs[int(key)]) 
    }
    outs.append(dict_element[element[key]])
  if len(outs)>1:
    added = tf.keras.layers.Add()(outs)
  else:
    added = outs[0]
  return added

def Cell_func(plan, channels, reduction, prev_reduction, x0, x1):
  preproc_1 = StdConv(channels, channels, 1, 1, "same", x1)
  preproc_0 = StdConv(channels, channels, 1, 1, "same", x0)
  if prev_reduction:
    preproc_0 = FactorizedReduce(channels, channels, x0)
  output = [preproc_0, preproc_1]
  for key in plan.keys():
    output.append(Node_func(plan[key], channels, reduction, output))
  if len(output)>3:
    out = tf.keras.layers.Concatenate(axis = -1)(output[2:])
  else:
    out = output[2]
  return out

def CNN_creation(channels, n_classes, n_layers, normal_plan, reduction_plan, input_shape, stem_multiplier=3):
  in_x = tf.keras.Input(input_shape)
  stem = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(channels*stem_multiplier, 3, strides=(1, 1), padding="same", data_format="channels_last"),
        tf.keras.layers.BatchNormalization(axis = -1)]
    )
  out_0 = out_1 = stem(in_x)
  red, prev_red = False, False
  cur_chans = channels
  for i in range(n_layers):
    prev_red = red
    red = i==n_layers//3 or i==2*n_layers//3
    plan = normal_plan
    if red :
      plan = reduction_plan
      cur_chans*=2
    out_0, out_1 = out_1, Cell_func(plan, cur_chans, red, prev_red, out_0, out_1)
  out = tf.keras.layers.GlobalAveragePooling2D()(out_1)
  out = tf.keras.layers.Flatten()(out)
  out = tf.keras.layers.Dense(n_classes, activation = "softmax")(out)
  return tf.keras.Model(inputs = in_x, outputs = out)