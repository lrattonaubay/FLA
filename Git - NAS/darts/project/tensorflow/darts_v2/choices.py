import tensorflow as tf
from typing import Dict

class LayerChoice(tf.Module):

    def __init__(self, candidates: Dict[str, tf.Module], label: str):
                 
        super(LayerChoice, self).__init__()
        self._label =label
        self._modules: Dict[str, tf.Module] = dict()

        self.names = []
        if isinstance(candidates, dict):
            for name, module in candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self._modules[name] = module
                self.names.append(name)
        else:
            raise TypeError("Unsupported candidates type: {}".format(type(candidates)))

        self.alpha = tf.Variable(tf.random.normal([len(self._modules)]) * 1e-3, name=self._label)

    @property
    def label(self):
        return self._label

    @property
    def choices(self):
        return  list(map(lambda name: [self._modules[name], name], self.names))

    def __call__(self, inputs):
        # Parcours chaque opération et la traverse avec les données 
        op_results = tf.stack([op(inputs) for op in self._modules.values()])
        alpha_shape = [op_results.get_shape()[0]] + [1] * (len(op_results.get_shape())-1)
        # Somme pour la sortie du noeud avec les poids appliqués
        return tf.math.reduce_sum(op_results * tf.reshape(tf.nn.softmax(self.alpha), alpha_shape), 0)

    def export(self):
        return list(self.names)[tf.math.argmax(self.alpha)]  


class InputChoice(tf.Module):

    def __init__(self, n_candidates: int, n_chosen: int, label: str):

        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self._label = label
        self.alpha = tf.Variable(tf.convert_to_tensor(tf.random.normal([self.n_candidates]) * 1e-3), name=self._label)

    @property
    def label(self):
        return self._label

    def __call__(self, inputs):
        inputs =  tf.stack(inputs)
        alpha_shape = [inputs.get_shape()[0]] + [1] * (len(inputs.get_shape()) - 1)
        return tf.math.reduce_sum(inputs * tf.reshape(tf.nn.softmax(self.alpha), alpha_shape), 0)

    def export(self):
        return tf.argsort(-self.alpha).numpy().tolist()[:self.n_chosen]    