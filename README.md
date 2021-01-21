
# HammingOne

## Goal:
Having a stet of long bit sequences find all pairs with the Hamming distance equal to 1.

## Implementation:

 - The program utilizes mutliple CUDA threads in order to parallelize the computation
 - In order to efficiently find the pairs a hash map was implemented, cuckoo hashing was used as it is the most suitable for GPU computations with multiple threads.
 ## How to use:
 1.  A generator of binary vectors can be found in /generator directory.  Using it one may generate a variable sized set of vectors of desired length. Each element of the set will have a pair with the Hamming distance equal to 1.
 2.  The main part of the program is HPU HammingOne solution that reads a file containing the vectors and counts the number of pairs with the Hamming distance equal to 1.