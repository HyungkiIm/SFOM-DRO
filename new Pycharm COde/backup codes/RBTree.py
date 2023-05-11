import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import time

"""
Oct-16
Fixed insertion, left and right rotation + printhelper.
Need to fix deletion function.


"""

# Node creation
class Node():
    def __init__(self, weight, index):
        self.weight = weight
        self.index = index
        self.nl = 0 #Number of nodes in left sub-tree
        self.nr = 0 #Number of nodes in right sub-tree
        self.sl = 0 #Sum of weights in left sub-tree
        self.sr = 0 #Sum of weights in right sub-tree
        self.parent = None
        self.left = None
        self.right = None
        self.color = 1 #1 is red


class RedBlackTree():
    def __init__(self):
        self.TNULL = Node(0,-1) #Tnull is like T.Nil
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL
        self.multi = 1
        self.addi = 0
        self.err_tol = 1e-10
    # # Preorder
    # def pre_order_helper(self, node):
    #     if node != TNULL:
    #         sys.stdout.write(node.item + " ")
    #         self.pre_order_helper(node.left)
    #         self.pre_order_helper(node.right)
    #
    # # Inorder
    # def in_order_helper(self, node):
    #     if node != TNULL:
    #         self.in_order_helper(node.left)
    #         sys.stdout.write(node.item + " ")
    #         self.in_order_helper(node.right)
    #
    # # Postorder
    # def post_order_helper(self, node):
    #     if node != TNULL:
    #         self.post_order_helper(node.left)
    #         self.post_order_helper(node.right)
    #         sys.stdout.write(node.item + " ")

    # Search the tree

    def construct_tree(self,size):
        weight = 1/size
        for i in range(size):
            self.insert(weight,i)

    def search_tree_helper(self, node, key):
        if node == self.TNULL or key == node.item:
            return node

        if key < node.item:
            return self.search_tree_helper(node.left, key)
        return self.search_tree_helper(node.right, key)

    # Balancing the tree after deletion
    def delete_fix(self, x):
        while x != self.root and x.color == 0:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    s = x.parent.right

                if s.left.color == 0 and s.right.color == 0:
                    s.color = 1
                    x = x.parent
                else:
                    if s.right.color == 0:
                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s = x.parent.right

                    s.color = x.parent.color
                    x.parent.color = 0
                    s.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    s = x.parent.left

                if s.right.color == 0 and s.left.color == 0:
                    s.color = 1
                    x = x.parent
                else:
                    if s.left.color == 0:
                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s = x.parent.left

                    s.color = x.parent.color
                    x.parent.color = 0
                    s.left.color = 0
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 0

    def __rb_transplant(self, u, v):

        if v != self.TNULL:
            if u.parent == None:
                self.root = v



            elif u == u.parent.left:
                u.parent.nl = v.nl + v.nr + 1
                u.parent.sl = v.sl + v.sr + v.weight
                u.parent.left = v

            else:
                u.parent.nr = v.nl + v.nr + 1
                u.parent.sr = v.sl + v.sr + v.weight
                u.parent.right = v

        else: # When v == self.TNULL. So we should update statistics of u.parent differently.

            if u.parent == None:
                self.root = v



            elif u == u.parent.left:
                u.parent.nl = 0
                u.parent.sl = 0
                u.parent.left = v

            else:
                u.parent.nr = 0
                u.parent.sr = 0
                u.parent.right = v

        v.parent = u.parent

    # Node deletion
    def delete_node_helper(self, node, weight, index):
        z = self.TNULL

        while node != self.TNULL:
            if node.weight == weight:
                if node.index == index:
                    z = node
                    break

                elif node.index < index:

                    node = node.right
                    if node != self.TNULL:
                        node.parent.nr -= 1
                        node.parent.sr -= weight

                elif node.index > index:
                    node = node.left
                    if node != self.TNULL:
                        node.parent.nl -= 1
                        node.parent.sl -= weight

            elif node.weight < weight:
                node = node.right
                if node != self.TNULL:
                    node.parent.nr -= 1
                    node.parent.sr -= weight

            elif node.weight > weight:
                node = node.left
                if node != self.TNULL:
                    node.parent.nl -= 1
                    node.parent.sl -= weight

        #Our code cannot handle this case.
        if z == self.TNULL:
            raise ValueError("Cannot find (weight,index) in the tree")
            return

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif (z.right == self.TNULL):
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == 0:
            self.delete_fix(x)

    # Balance the tree after insertion
    def fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right

                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    # Printing the tree
    def __print_helper(self, node, indent, last):
        if node != self.TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            s_color = "RED" if node.color == 1 else "BLACK"
            print('[(%s,%s), %s, %s, %s, %s](%s)' %(node.weight,node.index,node.nl,node.nr,node.sl,node.sr, s_color))
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    def sum_weight(self,node):
        if node != self.TNULL:
            return node.weight + self.sum_weight(node.left) + self.sum_weight(node.right)
        else:
            return 0

    def s_validation_helper(self,node):

        if abs(node.sl - self.sum_weight(node.left)) > self.err_tol:
            raise ValueError("Node (%s,%s) has wrong sl. sl of this node: %s, actual sum of left subtree: %s" \
                             %(node.weight,node.index, node.sl, self.sum_weight(node.left)))

        if node.left != self.TNULL:
            self.s_validation_helper(node.left)

        if abs(node.sr - self.sum_weight(node.right)) > self.err_tol:
            raise ValueError("Node (%s,%s) has wrong sr. sr of this node: %s, actual sum of right subtree: %s" \
                             %(node.weight,node.index, node.sr, self.sum_weight(node.right)))

        if node.right != self.TNULL:
            self.s_validation_helper(node.right)


    def s_validation(self):
        self.s_validation_helper(self.root)
        print("Sum Validation Completed.")


    def n_validation(self):
        self.n_validation_helper(self.root)
        print("Number Validation Completed.")



    def count_node(self,node):
        if node != self.TNULL:
            return 1 + self.count_node(node.left) + self.count_node(node.right)
        else:
            return 0

    def n_validation_helper(self,node):

        if  node.nl != self.count_node(node.left):
            raise ValueError("Node (%s,%s) has wrong nl. nl of this node: %s, actual node count of left subtree: %s" \
                             %(node.weight,node.index, node.nl, self.count_node(node.left)))

        if node.left != self.TNULL:
            self.n_validation_helper(node.left)

        if node.nr != self.count_node(node.right):
            raise ValueError("Node (%s,%s) has wrong nr. nr of this node: %s, actual node count of right subtree: %s" \
                             %(node.weight,node.index, node.nr, self.count_node(node.right)))

        if node.right != self.TNULL:
            self.n_validation_helper(node.right)










    def preorder(self):
        self.pre_order_helper(self.root)

    def inorder(self):
        self.in_order_helper(self.root)

    def postorder(self):
        self.post_order_helper(self.root)

    def searchTree(self, k):
        return self.search_tree_helper(self.root, k)

    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    def maximum(self, node):
        while node.right != self.TNULL:
            node = node.right
        return node

    def successor(self, x):
        if x.right != self.TNULL:
            return self.minimum(x.right)

        y = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    def predecessor(self,  x):
        if (x.left != self.TNULL):
            return self.maximum(x.left)

        y = x.parent
        while y != self.TNULL and x == y.left:
            x = y
            y = y.parent

        return y

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

        #update nl,nr,sl,sr of x and y
        x.nr = y.nl
        x.sr = y.sl

        y.nl = x.nl + y.nl + 1
        y.sl = x.sl + y.sl + x.weight


    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

        #update nl,nr,sl,sr of x and y
        x.nl = y.nr
        x.sl = y.sr
        y.nr = y.nr + x.nr + 1
        y.sr = y.sr + x.sr + x.weight

    def insert(self, weight, index):
        node = Node(weight,index)
        #node.parent = None
        #node.item = key
        node.left = self.TNULL
        node.right = self.TNULL
        #node.color = 1

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            if node.weight < x.weight:
                x.nl += 1
                x.sl += node.weight
                x = x.left
            elif node.weight > x.weight:
                x.nr += 1
                x.sr += node.weight
                x = x.right
            else: #When node.weight == x.weight
                if node.index < x.index:
                    x.nl += 1
                    x.sl += node.weight
                    x = x.left
                else:
                    x.nr += 1
                    x.sr += node.weight
                    x = x.right

        node.parent = y


        if y == None:
            self.root = node
        elif node.weight < y.weight:
            # y.nl += 1
            # y.sl += node.weight
            y.left = node
        elif node.weight > y.weight:
            # y.nr += 1
            # y.sr += node.weight
            y.right = node
        else: #node.weight == y.weight
            if node.index < y.index:
                # y.nl += 1
                # y.sl += node.weight
                y.left = node
            else:
                # y.nr += 1
                # y.sr += node.weight
                y.right = node

        if node.parent == None:
            node.color = 0
            return

        if node.parent.parent == None:
            return

        self.fix_insert(node)

    def get_root(self):
        return self.root

    def delete_node(self, weight, index):
        self.delete_node_helper(self.root, weight, index)

    def print_tree(self):
        self.__print_helper(self.root, "", True)

    def random_sample_helper(self, node, coin_arr):

        height = 0
        #How should we incorporate w_sum[i]?
        #Answer: We don't have to.

        if node == self.TNULL:
            raise ValueError("No node in tree. So cannot implement random sampling")

        while node.left != self.TNULL or node.right != self.TNULL: #If node is not a leaf node.
            #Instead of creating another random number, may be we can rescale coin
            coin = coin_arr[height]
            height +=1
            if node.left == self.TNULL: #this implies node.right != self.TNULL
                adj_sr = self.multi * node.sr + self.addi * node.nr
                adj_node_weight = self.multi * node.weight + self.addi
                if coin < adj_node_weight / (adj_node_weight + adj_sr):
                    return node.index
                else:
                    node = node.right
            else: #node.left != self.TNULL
                if node.right == self.TNULL:
                    adj_sl = self.multi * node.sl + self.addi * node.nl
                    adj_node_weight = self.multi * node.weight + self.addi
                    if coin < adj_node_weight / (adj_node_weight + adj_sl):
                        return node.index
                    else:
                        node = node.left
                else: #node has two children.
                    adj_sl = self.multi * node.sl + self.addi * node.nl
                    adj_node_weight = self.multi * node.weight + self.addi
                    adj_sr = self.multi * node.sr + self.addi * node.nr
                    if coin < adj_node_weight / (adj_node_weight + adj_sl + adj_sr):
                        return node.index
                    elif coin < (adj_node_weight + adj_sl) / (adj_node_weight + adj_sl + adj_sr):
                        node = node.left
                    else:
                        node = node.right



        return node.index







    def random_sample(self, coin_array):
        sample_list = []
        K_size, height= coin_array.shape
        for i in range(K_size):
            sample_list.append(int(self.random_sample_helper(self.root,coin_array[i,:])))

        return sample_list




if __name__ == "__main__":
    bst = RedBlackTree()

    n = 10000

    bst.construct_tree(n)

    # bst.insert(55)
    # bst.insert(40)
    # bst.insert(65)
    # bst.insert(60)
    # bst.insert(75)
    # bst.insert(57)

    bst.print_tree()

    print("\nStart Number Validation")
    bst.n_validation()
    print("\nStart Sum Validation with err_tol= %s" %bst.err_tol)
    bst.s_validation()

    #Deletion Verification

    # print("\nAfter deleting an element")
    # for i in range(950):
    #     bst.delete_node(1/n, i)
    # #Add 6 more
    # bst.insert(0.01, 0)
    # bst.insert(0.01, 1)
    # bst.insert(0.01, 2)
    # bst.insert(0.01, 3)
    # bst.insert(0.01, 4)
    # bst.insert(0.01, 5)
    #
    #
    # bst.print_tree()
    #
    # print("\nStart Number Validation")
    # bst.n_validation()
    # print("\nStart Sum Validation with err_tol= %s" % bst.err_tol)
    # bst.s_validation()

    #Random Sampling Verification

    sample_size = 500
    n_bins = n
    sample_list = bst.random_sample(sample_size)

    unique, counts = np.unique(sample_list, return_counts=True)
    print(dict(zip(unique, counts)))

    plt.hist(sample_list, n_bins)
    plt.show()

