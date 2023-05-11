import numpy as np


class BSTNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def add_child(self, data):
        if data == self.data:
            return
        if data< self.data:
            #add data in left subtree
            if self.left: #not leaf node
                self.left.add_child(data)
            else:
                self.left = BSTNode(data)

        else:
            #add data in right subtree
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BSTNode(data)

    def in_order_traversal(self):
        elements = []
        #visit left tree
        if self.left:
            elements += self.left.in_order_traversal()
        #visit base node
        elements.append(self.data)
        #visit right node
        if self.right:
            elements += self.right.in_order_traversal()

        return elements
    def search(self,val):
        if self.data == val:
            return True
        if val<self.data:
            #val might be in the left subtree
            if self.left:
               return self.left.search(val)
            else:
                return False
        if val>self.data:
            #val might be in the right subtree
            if self.right:
                return self.right.search(val)
            else:
                return False

    def find_max(self):
        if self.right is None:
            return self.data
        else:
            return self.right.find_max()

    def find_min(self):
        if self.left is None:
            return self.data
        else:
            return self.left.find_min()

    def delete_node(self,val):
        if val<self.data:
            if self.left:
                self.left = self.left.delete_node(val)
        elif val>self.data:
            if self.right:
                self.right = self.right.delete_node(val)
        else:
            if self.left is None and self.right is None:
                return None
            if self.left is None:
                return self.right
            if self.right is None:
                return self.left
            min_val = self.right.find_min()
            self.data = min_val
            self.right = self.right.delete_node(min_val)
        return self



def build_tree(elements):
    root = BSTNode(elements[0])

    for i in range(1,len(elements)):
        root.add_child(elements[i])

    return root

if __name__ == '__main__':
    numbers = [17,4,1,20,9,23,18,24]
    numbers_tree = build_tree(numbers)
    numbers_tree = numbers_tree.delete_node(20)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(9)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(24)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(1)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(4)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(17)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(18)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(23)
    print(numbers_tree.in_order_traversal())
    numbers_tree = numbers_tree.delete_node(18)

