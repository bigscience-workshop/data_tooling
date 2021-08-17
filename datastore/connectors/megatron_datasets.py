import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

#################################################################################################################################
# Code for megatron mmap indexed datasets, which are in turn based on memmap datasets, and a custom numpy based index file.

