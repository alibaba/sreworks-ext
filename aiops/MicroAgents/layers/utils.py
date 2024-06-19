from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union
import networkx as nx
from collections import deque
from functools import partial



def get_neighbors( node, G=None):
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    return predecessors + successors

    


def get_k_order_neighbors_with_direction(G, node, k, direction:Literal['forward', 'backward', 'bidirectional']='forward'):
    if k < 0:
        raise ValueError("Order k must be non-negative")

    # 用于保存已访问的节点，防止重复计算
    visited = set()
    visited.add(node)

    # 双端队列保存节点和当前跳数
    queue = deque([(node, 0)])

    # 用字典按跳数保存后继节点
    neighbors_by_order = {i: set() for i in range(1, k+1)}

    if direction == 'bidirectional':
        direction_func = partial(get_neighbors, G=G)
    else:
        direction_func = G.successors if direction == 'forward' else G.predecessors
    # 在队列中进行BFS
    while queue:
        current_node, jumps = queue.popleft()

        # 如果跳数达到K，则停止向队列中添加这个节点的子节点
        if jumps < k:
            for neighbor in direction_func(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, jumps + 1))
                    # 只有当跳数小于K时我们才将其视为该阶的后继节点
                    if jumps + 1 <= k:
                        neighbors_by_order[jumps + 1].add(neighbor)

    return neighbors_by_order