from networkx import Graph

from discopy.sugar import dataclass
from discopy.monoidal import Ty, Diagram, Box, Encoding


@dataclass
class Node:
    kind: str
    label: object  #: Ty | Box
    i: int
    j: int


def diagram2graph(diagram: Diagram) -> Graph:
    graph = Graph()
    scan = [Node('dom', x, i, -1) for i, x in enumerate(diagram.dom)]
    graph.add_edges_from(zip(scan, scan))
    for j, (left, box, _) in enumerate(diagram.inside):
        box_node = Node('box', box, -1, j)
        dom_nodes = [Node('dom', x, i, j) for i, x in enumerate(box.dom)]
        cod_nodes = [Node('cod', x, i, j) for i, x in enumerate(box.cod)]
        graph.add_edges_from(zip(scan[len(left): len(left @ box.dom)], dom_nodes))
        graph.add_edges_from(zip(dom_nodes, len(box.dom) * [box_node]))
        graph.add_edges_from(zip(len(box.cod) * [box_node], cod_nodes))
        scan = scan[:len(left)] + cod_nodes + scan[len(left @ box.dom):]
    graph.add_edges_from(zip(scan, [
        Node('cod', x, i, len(diagram)) for i, x in enumerate(diagram.cod)]))
    return graph


Embedding = dict[Node, tuple[float, float]]
PlaneGraph = tuple[Graph, Embedding]


def make_space(pos: Embedding, scan: list[Node], box: Box, off: int
        ) -> tuple[Embedding, float]:
    if not scan:
        return pos, 0
    half_width = len(box.cod[:-1]) / 2 + 1
    if not box.dom:
        if not off:
            x_pos = pos[scan[0]][0] - half_width
        elif off == len(scan):
            x_pos = pos[scan[-1]][0] + half_width
        else:
            right = pos[scan[off + len(box.dom)]][0]
            x_pos = (pos[scan[off - 1]][0] + right) / 2
    else:
        right = pos[scan[off + len(box.dom) - 1]][0]
        x_pos = (pos[scan[off]][0] + right) / 2
    if off and pos[scan[off - 1]][0] > x_pos - half_width:
        limit = pos[scan[off - 1]][0]
        pad = limit - x_pos + half_width
        for node, position in pos.items():
            if position[0] <= limit:
                pos[node] = (pos[node][0] - pad, pos[node][1])
    if off + len(box.dom) < len(scan)\
            and pos[scan[off + len(box.dom)]][0] < x_pos + half_width:
        limit = pos[scan[off + len(box.dom)]][0]
        pad = x_pos + half_width - limit
        for node, position in pos.items():
            if position[0] >= limit:
                pos[node] = (pos[node][0] + pad, pos[node][1])
    return pos, x_pos


def draw(self: Diagram) -> PlaneGraph:
    graph = diagram2graph(self)
    box_nodes = [Node('box', box, -1, j) for j, box in enumerate(self.boxes)]
    dom_nodes = scan = [Node('dom', x, i, -1) for i, x in enumerate(self.dom)]
    position = {node: (i, -1) for i, node in enumerate(dom_nodes)}
    for j, (left, box, _) in enumerate(self.inside):
        box_node = Node('box', box, -1, j)
        position, left_of_box = make_space(position, scan, box, len(left))
        position[box_node] = (
            left_of_box + max(len(box.dom), len(box.cod)) / 2, j)
        for i, x in enumerate(box.dom):
            cod_node, = filter(lambda node: node.kind != "box",
                               graph.neighbors(Node('dom', x, i, j)))
            position[Node('dom', x, i, j)] = (position[cod_node][0], j - .1)
        for i, x in enumerate(box.cod):
            position[Node('cod', x, i, j)] = (left_of_box + i, j + .1)
        box_cod_nodes = [Node('cod', x, i, j) for i, x in enumerate(box.cod)]
        scan = scan[:len(left)] + box_cod_nodes + scan[len(left @ box.dom):]
    for i, x in enumerate(self.cod):
        cod_node = Node('cod', x, i, len(self))
        position[cod_node] = (position[scan[i]][0], len(self))
    return graph, position


def graph2diagram(graph: Graph, position: Embedding) -> Diagram:
    dom = Ty().tensor(*[node.label for node in graph.nodes
                        if node.kind == 'dom' and node.j == -1])
    boxes = [node.label for node in graph.nodes if node.kind == 'box']
    scan, offsets = [Node('dom', x, i, -1) for i, x in enumerate(dom)], []
    for j, box in enumerate(boxes):
        left_of_box = position[Node('dom', box.dom[0], 0, j)][0]\
            if box.dom else position[Node('box', box, -1, j)][0]
        offset = len([node for node in scan if position[node][0] < left_of_box])
        box_cod_nodes = [Node('cod', x, i, j) for i, x in enumerate(box.cod)]
        scan = scan[:offset] + box_cod_nodes + scan[offset + len(box.dom):]
        offsets.append(offset)
    return Diagram.decode(Encoding(dom, list(zip(boxes, offsets))))
