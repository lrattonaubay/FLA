from utils import split_normal_reduce, convert_to_simple
from graphviz import Digraph



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