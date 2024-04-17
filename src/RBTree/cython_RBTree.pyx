from libc.stdio cimport printf, snprintf
import numpy as np
cimport numpy as cnp
import cython
from multiprocessing import Pool

cdef class Node:
    cdef double weight
    cdef int index, nl, nr, color
    cdef double sl, sr
    cdef Node parent, left, right

    def __cinit__(self, double weight, int index):
        self.weight = weight
        self.index = index
        self.nl = 0  # Number of nodes in left sub-tree
        self.nr = 0  # Number of nodes in right sub-tree
        self.sl = 0.0  # Sum of weights in left sub-tree
        self.sr = 0.0  # Sum of weights in right sub-tree
        self.parent = None
        self.left = None
        self.right = None
        self.color = 1  # 1 is red

cdef class RedBlackTree:
    cdef Node TNULL, root
    cdef double _multi, _addi
    cdef double err_tol

    def __cinit__(self):
        self.TNULL = Node(0, -1)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL
        self._multi = 1.0
        self._addi = 0.0
        self.err_tol = 1e-10

    cdef void insert(self, double weight, int index):
        cdef Node node = Node(weight, index)
        node.left = self.TNULL
        node.right = self.TNULL

        cdef Node y = None
        cdef Node x = self.root

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
            y.left = node
        elif node.weight > y.weight:
            y.right = node
        else: #node.weight == y.weight
            if node.index < y.index:
                y.left = node
            else:
                y.right = node

        if node.parent == None:
            node.color = 0
            return

        if node.parent.parent == None:
            return

        self.fix_insert(node)

    cdef void fix_insert(self, Node k):
        cdef Node u
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

    def insert_node(self, double weight, int index):
        self.insert(weight, index)

    cdef void left_rotate(self, Node x):
        cdef Node y = x.right
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

    cdef void right_rotate(self, Node x):
        cdef Node y = x.left
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

    cdef void construct_tree(self, int size):
        cdef double weight = 1.0 / size
        cdef int i
        for i in range(size):
            self.insert(weight, i)

    def construct_tree_py(self, int size):
        self.construct_tree(size)

    cpdef void print_tree(self):
        self._print_helper(self.root, b"", 1)

    cdef void _print_helper(self, Node node, const char* indent, bint last):
        cdef const char* s_color
        cdef char new_indent[100]

        if node != self.TNULL:
            printf("%s", indent)

            if last:
                printf("R----")
                snprintf(new_indent, sizeof(new_indent), b"%s     ", indent)
            else:
                printf("L----")
                snprintf(new_indent, sizeof(new_indent), b"%s|    ", indent)

            s_color = "RED" if node.color == 1 else "BLACK"
            printf("[(%.2f, %d), %d, %d, %.2f, %.2f](%s)\n",
                   node.weight, node.index, node.nl, node.nr, node.sl, node.sr, s_color)
            self._print_helper(node.left, new_indent, 0)
            self._print_helper(node.right, new_indent, 1)

    cdef double sum_weight(self, Node node):
        if node != self.TNULL:
            return node.weight + self.sum_weight(node.left) + self.sum_weight(node.right)
        else:
            return 0.0

    cdef int count_node(self, Node node):
        if node != self.TNULL:
            return 1 + self.count_node(node.left) + self.count_node(node.right)
        else:
            return 0

    def get_node_count(self):
        return self.count_node(self.root)

    cdef Node minimum(self, Node node):
        while node.left != self.TNULL:
            node = node.left
        return node

    cdef Node maximum(self, Node node):
        while node.right != self.TNULL:
            node = node.right
        return node

    cdef Node successor(self, Node x):
        cdef Node y
        if x.right != self.TNULL:
            return self.minimum(x.right)

        y = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    cdef Node predecessor(self, Node x):
        cdef Node y
        if x.left != self.TNULL:
            return self.maximum(x.left)

        y = x.parent
        while y != self.TNULL and x == y.left:
            x = y
            y = y.parent
        return y

    cpdef void delete_node(self, double weight, int index):
        self.delete_node_helper(self.root, weight, index)

    cdef void delete_node_helper(self, Node node, double weight, int index):
        cdef Node z = self.TNULL

        while node != self.TNULL:
            if node.weight == weight and node.index == index:
                z = node
                break
            elif node.weight < weight or (node.weight == weight and node.index < index):
                node = node.right
                if node != self.TNULL:
                    node.parent.nr -= 1
                    node.parent.sr -= weight
            else:
                node = node.left
                if node != self.TNULL:
                    node.parent.nl -= 1
                    node.parent.sl -= weight
        self.__delete_node_actual(z)

    cdef void __delete_node_actual(self, Node z):
        cdef Node x, y
        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif z.right == self.TNULL:
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
                self.__rb_transplant(z, y)
                y.left = z.left
                y.left.parent = y
                y.color = z.color
                # update nl sl of y
                y.nl = z.nl
                y.sl = z.sl
            else:
                self.__rb_transplant(y, y.right)
                # update y.parent's nl and sl
                y.parent.nl = y.nr
                y.parent.sl = y.sr

                y.right = z.right
                y.right.parent = y
                # update y's nr and sr
                y.nr = z.nr - 1
                y.sr = z.sr - y.weight

                self.__rb_transplant(z, y)
                y.left = z.left
                y.left.parent = y
                # update y's nl and sl
                y.nl = z.nl
                y.sl = z.sl
                y.color = z.color

        if y_original_color == 0:
            self.delete_fix(x)

    cdef void delete_fix(self, Node x):
        cdef Node s
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

    cdef void __rb_transplant(self, Node u, Node v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int random_sample_helper(self, Node node, double[:] coin_arr):
        cdef:
            int height = 0
            double adj_sr, adj_node_weight, adj_sl, coin

        while node.left != self.TNULL or node.right != self.TNULL:
            coin = coin_arr[height]
            height +=1
            if node.left == self.TNULL:
                adj_sr = self._multi * node.sr + self._addi * node.nr
                adj_node_weight = self._multi * node.weight + self._addi
                if coin < adj_node_weight / (adj_node_weight + adj_sr):
                    return node.index
                else:
                    node = node.right
            else:
                if node.right == self.TNULL:
                    adj_sl = self._multi * node.sl + self._addi * node.nl
                    adj_node_weight = self._multi * node.weight + self._addi
                    if coin < adj_node_weight / (adj_node_weight + adj_sl):
                        return node.index
                    else:
                        node = node.left
                else:
                    adj_sl = self._multi * node.sl + self._addi * node.nl
                    adj_node_weight = self._multi * node.weight + self._addi
                    adj_sr = self._multi * node.sr + self._addi * node.nr
                    if coin < adj_node_weight / (adj_node_weight + adj_sl + adj_sr):
                        return node.index
                    elif coin < (adj_node_weight + adj_sl) / (adj_node_weight + adj_sl + adj_sr):
                        node = node.left
                    else:
                        node = node.right
        return node.index

    @cython.boundscheck(False)
    @cython.wraparound(False)

    cdef list _random_sample(self, double[:, :] coin_array):
        cdef:
            int i
            list sample_list = []

        cdef int K_size = coin_array.shape[0]

        # Loop over each set of arguments and apply the helper directly
        for i in range(K_size):
            # Directly call the helper function without using multiprocessing
            sample_list.append(self.random_sample_helper(self.root, coin_array[i, :]))

        return sample_list

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def random_sample(self, coin_array):
        # ensure coin_array is a NumPy array with dtype float64 and C-contiguous
        cdef double[:, :] c_coin_array = np.ascontiguousarray(coin_array, dtype=np.float64)
        return self._random_sample(c_coin_array)


    cdef cnp.ndarray[cnp.int_t, ndim=2, mode="c"] _random_sample_all(self, double[:,:,:] coin_K_arr):
        cdef:
            int i, k
            int sample_freq = coin_K_arr.shape[0]
            int K = coin_K_arr.shape[1]
            cnp.ndarray[cnp.int_t, ndim=2, mode="c"] sample_list

        # Creating a C-contiguous D int array for optimized access in C space
        sample_list = np.empty((sample_freq, K), dtype=np.intc, order='C')

        for i in range(sample_freq):
            for k in range(K):
                sample_list[i, k] = self.random_sample_helper(self.root, coin_K_arr[i, k, :])

        return sample_list

    def random_sample_all(self, coin_K_arr):
        # Ensure coin_K_arr is a NumPy array with dtype float64 and C-contiguous
        cdef double[:,:,:] c_coin_K_arr = np.ascontiguousarray(coin_K_arr, dtype=np.float64)
        return self._random_sample_all(c_coin_K_arr)


    @property
    def multi(self):
        return self._multi

    @multi.setter
    def multi(self, double value):
        self._multi = value

    @property
    def addi(self):
        return self._addi

    @addi.setter
    def addi(self, double value):
        self._addi = value