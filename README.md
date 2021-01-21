# HammingOne
GPU CUDA hash map implementation for counting number of vector pairs with Hamming distance equal to 1.

The program contains a cpu generator (/Generator/generator.cpu) which can be used to generate an arbitrary number of binary vectors
It should be called: generator N L (N - number of vectors, L - length of vectors).

The main part of the the program is GPU HammingOne solution reading a file and counting number of pairs with Hamming distance equal to 1.
The result is achieved by cuckoo hashing all the vectors and then checking whether each possible pairs is stored in the hash table.


