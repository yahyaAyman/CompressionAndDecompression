"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes

def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """

    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    frequency_dictionary = {}
    
    for element in text:
            if not(element in frequency_dictionary):
                frequency_dictionary[element] = 1
            else:
                frequency_dictionary[element] += 1

    return frequency_dictionary



def fix_tree(freq_dict, tree):
    if not tree:
        return None
    
    if tree.right != None:
        tree.symbol = None
        
    if tree.left != None:
        tree.symbol = None
        
    if tree.right != None and tree.left != None:
        tree.symbol = None
        
    fix_tree(freq_dict, tree.right)
    fix_tree(freq_dict, tree.left)

    return tree
    
def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    if len(freq_dict) == 0:
        return None
    
    if len(freq_dict) == 1:
        for index in freq_dict:
           return HuffmanNode(index)

    sorted_lst = []
    for i in freq_dict:
        sorted_lst.append((freq_dict[i], HuffmanNode(i)))

    #The code used to sort the list was taken from the following link
    #https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
    sorted_lst = sorted(sorted_lst, key=lambda x: x[0], reverse=False)
    while len(sorted_lst) != 2:
        #This below line of code was take from
        #https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
        sorted_lst = sorted(sorted_lst, key=lambda x: x[0], reverse=False)
        combined_tree = HuffmanNode((sorted_lst[0][0] + sorted_lst[1][0]), sorted_lst[0][1], sorted_lst[1][1])
        sorted_lst.append((sorted_lst[0][0] + sorted_lst[1][0] , combined_tree))
        sorted_lst = (sorted_lst[2:])
        
    #This below line of code was take from
    #https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
    sorted_lst = sorted(sorted_lst, key=lambda x: x[0], reverse=False)
    combined_tree = HuffmanNode((sorted_lst[0][0] + sorted_lst[1][0]), sorted_lst[0][1], sorted_lst[1][1])
    sorted_lst.append((sorted_lst[0][0] + sorted_lst[1][0] ,combined_tree))

    return fix_tree(freq_dict, combined_tree)


def get_codes_helper(tree, temp_dict, total):
    if not tree:
        return None
    
    if tree.symbol != None and tree.right == None and tree.left == None:
        temp_dict[tree.symbol] = total

    if tree.symbol == None:
        if tree.right:
            get_codes_helper(tree.right, temp_dict, total +'1')
            
        if tree.left:
            get_codes_helper(tree.left, temp_dict, total+'0')

    return temp_dict


def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    return get_codes_helper(tree, {}, '')

#This part of code was taken from Professor Dan
def postorder(tree):
    lst = []
    if tree:
        lst += (postorder(tree.left) + postorder(tree.right) + [tree])
    return lst
    

def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    lst_of_node = postorder(tree)
    counter = 0
    for element in lst_of_node:
        if element.symbol == None:
            element.number = counter
            counter += 1



def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    sum_freq = []
    tree_codes = get_codes(tree)
    for index in tree_codes:
        value = len(tree_codes[index]) * freq_dict[index]
        sum_freq.append(value)
    sum_freq = sum(sum_freq) #sum of freqs multiplied by the length of each bit


    lst_freq = []
    for index in freq_dict:
        value = freq_dict[index]
        lst_freq.append(value)

    lst_freq = sum(lst_freq) #sum of freqs

    return (sum_freq / lst_freq)
    

def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    string = ''
    lst = []
    for item in text:
        string += codes[item]
        if len(string) > 8:
            lst.append(bits_to_byte(string[:8]))
            string = string[8:]

    if len(string) != 0:
        if len(string) == 8:
            lst.append(bits_to_byte(string))
            return bytes(lst)
        
        if len(string) < 8:
            needed_num = 8 - len(string)
            missing_piece = '0' * needed_num
            string += missing_piece
            data = bits_to_byte(string)
            lst.append(data)

    return bytes(lst)

def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """    
    lst_of_vals = []
    p_ordered_lst = postorder(tree)    
    for element in p_ordered_lst:
        if element.left != None or element.right != None:        
                if element.left.left == None and element.left.right == None: #########################What does this do?
                    lst_of_vals.append(0)
                    lst_of_vals.append(element.left.symbol)
                else:
                    lst_of_vals.append(1)
                    lst_of_vals.append(element.left.number)

                if element.right.left == None and element.right.right == None:
                    lst_of_vals.append(0)
                    lst_of_vals.append(element.right.symbol)
                else:
                    lst_of_vals.append(1)
                    lst_of_vals.append(element.right.number)

    return bytes(lst_of_vals)



def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    tree = HuffmanNode(None)
    if len(node_lst) == 0:
        return
    
    current = node_lst[root_index]

    if current.l_type == 0 and current.r_type == 0:
        tree.left = HuffmanNode(current.l_data)
        tree.right = HuffmanNode(current.r_data)

    if current.l_type == 0:
        tree.left = HuffmanNode(current.l_data)
    if current.r_type == 0:
        tree.right = HuffmanNode(current.r_data)

    if current.l_type == 1:
         tree.left = generate_tree_general(node_lst, current.l_data)

    if current.r_type == 1:
        tree.right = generate_tree_general(node_lst, current.r_data)

    return tree


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """

    tree = HuffmanNode(None)

    if node_lst[root_index].r_type == 1:
        right_index = root_index - 1
        tree.right = generate_tree_postorder(node_lst, right_index)
    else:
        tree.right = HuffmanNode(node_lst[root_index].r_data)
        
    if node_lst[root_index].l_type == 1:
        left_index = root_index - 2
        tree.left = generate_tree_postorder(node_lst, left_index)
    else:
        tree.left = HuffmanNode(node_lst[root_index].l_data)

    return tree



def swap_key_val(dic):
    temp = {}
    for index in dic:
        temp[dic[index]] = index

    return temp    
    
def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """
    codes = get_codes(tree)
    swapped_codes = swap_key_val(codes)
    lst = []
    final = ''
    temp = ''
    counter = 0
    
    temp_resut = [byte_to_bits(byte) for byte in text]
    for i in temp_resut:
        final += i


    for i in final:
        if counter <= size:
            temp += i 
            if temp in swapped_codes:
                lst.append(swapped_codes[temp])
                temp = ''
                counter += 1
    return bytes(lst)
        
def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    

    sorted_lst = order(freq_dict)#Recursive call to sort the list
    lst_of_nodes = [tree]#We are using a lis to keep track of the nodes
    
    while len(lst_of_nodes) != 0:
        curr_node = lst_of_nodes.pop(0)
        if curr_node.right == None and curr_node.left == None:
            element = sorted_lst.pop(len(sorted_lst) - 1)
            curr_node.symbol = element[1]
            
        if curr_node.left != None:
            lst_of_nodes.append(curr_node.left)

        if curr_node.right != None:
            lst_of_nodes.append(curr_node.right)


def order(dic):
    final_lst = [] #Appending all the 

    for i in dic:
        final_lst.append((dic[i], i))
    
    return sorted(final_lst)

if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
