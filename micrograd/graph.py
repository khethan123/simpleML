from graphviz import Digraph

def trace(root):
    """
    Builds a set of all nodes and edges in a graph.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_graph(root, format='png', rankdir='TB'):
    """
    Draws a graph of the computation using Graphviz.

    format: str, format of the output file (e.g. 'png', 'svg')
    rankdir: str, direction of the graph (e.g. 'TB', 'LR')
    """
    assert rankdir in ['LR', 'TB']
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}", shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot