import choices

def _replace_module_with_type(root_module, type_name, modules):
    if modules is None:
        modules = []

    for child in root_module.submodules:
        if isinstance(child, type_name):
            modules.append([child.label, child])
            
    return modules


def replace_layer_choice(root_module, modules):

    return _replace_module_with_type(root_module, choices.LayerChoice, modules)


def replace_input_choice(root_module, modules):

    return _replace_module_with_type(root_module, choices.InputChoice, modules)

