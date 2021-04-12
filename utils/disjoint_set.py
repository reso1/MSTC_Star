"""
DisjointSet Implmentation
ref: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
"""


class DisJointSetNode:
    def __init__(self, data, parent=None, rank=0, size=1):
        self.data = data
        self.parent = parent
        self.rank = rank
        self.size = size

    def __eq__(self, other):
        return self.data == other.data


class DisjointSet:
    def __init__(self):
        self.trees = []

    def make(self, obj) -> DisJointSetNode:
        # under the assertion that obj is not in the DisjointSetForest
        # don't call make_set to identical obj

        node = DisJointSetNode(obj)
        node.parent = node

        self.trees.append([node])
        return node

    def find(self, node: DisJointSetNode) -> DisJointSetNode:
        """ path compression Find OP """
        # find root of node
        root = node
        while root.parent != root:
            root = root.parent

        # assign root to parent of every node in path(node->root)
        while node.parent != root:
            parent = node.parent
            node.parent = root
            node = parent

        return root

    def union(self, p: DisJointSetNode, q: DisJointSetNode):
        """ Union by rank """
        root_p, root_q = self.find(p), self.find(q)

        # already in the same set
        if root_p == root_q:
            return

        # swap root of p and root of q
        if root_p.rank < root_q.rank:
            root_p, root_q = root_q, root_p

        # merge root_p and root_q
        root_q.parent = root_p
        if root_p.rank == root_q.rank:
            root_p.rank += 1
