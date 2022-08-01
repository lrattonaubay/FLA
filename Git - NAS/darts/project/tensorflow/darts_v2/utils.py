from graphviz import Digraph

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

def visualize(final_architecture):
    arc_normal, arc_reduce = split_normal_reduce(final_architecture)
    values_normal = convert_to_simple(arc_normal)
    values_reduce = convert_to_simple(arc_reduce)

    if values_normal and len(values_normal)>0:
        normal_graph = generate_graph(values_normal, 'normal')
        normal_graph.render(directory='img', view=False)
    
    if values_reduce and len(values_reduce)>0:
        reduce_graph = generate_graph(values_reduce, 'reduce')
        reduce_graph.render(directory='img', view=False)

def generate_graph(model, name):

    graph = Digraph(name)

    graph.node('0', 'c[k-2]')
    graph.node('1', 'c[k-1]')

    for i in range(len(model)) :
        graph.node(str(i+2),str(i))

    graph.node(str(len(model)+2), 'c[k]')

    for key, values in model.items() :
        for node, value in values.items() :
            graph.edge(str(node), str(key), label=str(value))
    
    graph.edge(str(len(model)+1),str(len(model)+2))

    return graph


def split_normal_reduce(arc):
    """
    Description
    ---------------
    Splits the models into normal and reduce dictionaries.
    Input(s)
    ---------------
    arc: dictionary
    Output(s)
    ---------------
    arc_normal: dictionary
    arc_reduce: dictionary
    """
    arc_normal, arc_reduce = dict(),dict()
    for key in arc.keys():
        if "normal" in key:
            arc_normal[key]=arc[key]
        elif "reduce" in key:
            arc_reduce[key]=arc[key]
        else:
            print("Issue encountered : the following value is neither", key)
    return arc_normal, arc_reduce

def convert_to_simple(arc):
    """
    Description
    ---------------
    Converts an architecture into a more understandable dictionary.
    Output shape : {n:{p:<type of link>}}
    Output example : {2: {1: 'maxpool', 0: 'maxpool'}, 3: {2: 'maxpool', 1: 'maxpool'}, 4: {3: 'maxpool', 2: 'maxpool'}, 5: {3: 'maxpool', 4: 'maxpool'}}
    Input(s)
    ---------------
    arc: dictionary
    Output(s)
    ---------------
    kept_arc: dictionary
    """
    kept_arc_index = []
    kept_arc = dict()
    for value in arc.values():
        if type(value)==type([1,2]):
            kept_arc_index.append(value)
    prev_inc = 0
    increment = 2
    j=2
    for pair in kept_arc_index:
        keys_available = list(arc.keys())[prev_inc:increment]
        values = dict()
        for i in pair:
            values[i]= arc[keys_available[i]]
        kept_arc[j] = values
        prev_inc= increment
        increment += j+1
        j+=1
    
    return kept_arc
