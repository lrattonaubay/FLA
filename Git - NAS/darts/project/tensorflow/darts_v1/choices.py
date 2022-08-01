import tensorflow as tf
from typing import Union, Dict, List, Optional, OrderedDict
import warnings
import choices_utils

class LayerChoice(tf.Module):

    def create_fixed_module(cls, candidates: Union[Dict[str, tf.Module], List[tf.Module]], *,
                            label: Optional[str] = None, **kwargs):
        chosen = choices_utils.get_fixed_value(label)
        if isinstance(candidates, list):
            return candidates[int(chosen)]
        else:
            return candidates[chosen]

    def __init__(self, candidates: Union[Dict[str, tf.Module], List[tf.Module]], *,
                 prior: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
                 
        super(LayerChoice, self).__init__()
        if 'key' in kwargs:
            warnings.warn(f'"key" is deprecated. Assuming label.')
            label = kwargs['key']
        if 'return_mask' in kwargs:
            warnings.warn(f'"return_mask" is deprecated. Ignoring...')
        if 'reduction' in kwargs:
            warnings.warn(f'"reduction" is deprecated. Ignoring...')
        self.candidates = candidates
        self.prior = prior or [1 / len(candidates) for _ in range(len(candidates))]
        assert abs(sum(self.prior) - 1) < 1e-5, 'Sum of prior distribution is not 1.'
        self._label = choices_utils.generate_new_label(label)
        self._modules: Dict[str, Optional[tf.Module]] = OrderedDict()

        self.names = []
        if isinstance(candidates, dict):
            for name, module in candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                self._modules[name] = module
                self.names.append(name)
        elif isinstance(candidates, list):
            for i, module in enumerate(candidates):
                self._modules[name] = module
                self.names.append(str(i))
        else:
            raise TypeError("Unsupported candidates type: {}".format(type(candidates)))
        self._first_module = self._modules[self.names[0]]  # to make the dummy forward meaningful

    @property
    def key(self):
        return self._key()

    def _key(self):
        warnings.warn('Using key to access the identifier of LayerChoice is deprecated. Please use label instead.',
                      category=DeprecationWarning)
        return self._label

    @property
    def label(self):
        return self._label

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]

    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    @property
    def choices(self):
        return self._choices()

    def _choices(self):
        warnings.warn("layer_choice.choices is deprecated. Use `list(layer_choice)` instead.", category=DeprecationWarning)
        return list(self)

    def __call__(self, x):
        #warnings.warn('You should not run forward of this module directly.')
        return self._first_module(x)

    def __repr__(self):
        return f'LayerChoice({self.candidates}, label={repr(self.label)})'


class InputChoice(tf.Module):
  
    @classmethod
    def create_fixed_module(cls, n_candidates: int, n_chosen: Optional[int] = 1, reduction: str = 'sum', *,
                            prior: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
        return ChosenInputs(choices_utils.get_fixed_value(label), reduction=reduction)

    def __init__(self, n_candidates: int, n_chosen: Optional[int] = 1,
                 reduction: str = 'sum', *,
                 prior: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
        super(InputChoice, self).__init__()
        if 'key' in kwargs:
            warnings.warn(f'"key" is deprecated. Assuming label.')
            label = kwargs['key']
        if 'return_mask' in kwargs:
            warnings.warn(f'"return_mask" is deprecated. Ignoring...')
        if 'choose_from' in kwargs:
            warnings.warn(f'"reduction" is deprecated. Ignoring...')
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.prior = prior or [1 / n_candidates for _ in range(n_candidates)]
        assert self.reduction in ['mean', 'concat', 'sum', 'none']
        self._label = choices_utils.generate_new_label(label)

    @property
    def key(self):
        return self._key()

    def _key(self):
        warnings.warn('Using key to access the identifier of InputChoice is deprecated. Please use label instead.',
                      category=DeprecationWarning)
        return self._label

    @property
    def label(self):
        return self._label

    def __call__(self, candidate_inputs: List[tf.Tensor]) -> tf.Tensor:
        #warnings.warn('You should not run forward of this module directly.')
        return candidate_inputs[0]

    def __repr__(self):
        return f'InputChoice(n_candidates={self.n_candidates}, n_chosen={self.n_chosen}, ' \
            f'reduction={repr(self.reduction)}, label={repr(self.label)})'


class ChosenInputs(tf.Module):
    """
    A module that chooses from a tensor list and outputs a reduced tensor.
    The already-chosen version of InputChoice.
    """

    def __init__(self, chosen: Union[List[int], int], reduction: str):
        super().__init__()
        self.chosen = chosen if isinstance(chosen, list) else [chosen]
        self.reduction = reduction

    def __call__(self, candidate_inputs):
        return self._tensor_reduction(self.reduction, [candidate_inputs[i] for i in self.chosen])

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == 'none':
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == 'sum':
            return sum(tensor_list)
        if reduction_type == 'mean':
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == 'concat':
            return tf.cat(tensor_list, dim=1)
        raise ValueError(f'Unrecognized reduction policy: "{reduction_type}"')