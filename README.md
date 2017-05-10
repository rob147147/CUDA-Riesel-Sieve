This is a sieve for eliminating non-primes of the form k*b^n-1, typically searched by the Conjectures R'Us (CRUS) project over at mersenneforum.org.

The CPU is responsible for reading in ABCD files and generating arrays of prime numbers to send to the GPU.
The GPU takes an array of prime numbers which may be prime factors of numbers of the form k*b^n-1. The GPU is also given a list of k-values and the n-min and n-max value from the ABCD file. 

The core algorithm run on the GPU is the Baby Steps Giants Steps (BSGS) algorithm for solving the discrete logarithm. 


This is very much a work in progress but these are some current performance numbers from CUDA enabled NVidia GPUs to which I have access. The test file is R745.ABCD which contains 22 k-values, with n-min = 180,000 and n-max = 250,000 giving an n-range of 70,000.

CPU (1 core of i5-4440 @ 3.1Ghz) - 6,000,000 p/sec \n
GeForce GT 710 (366 GFLOPS, 19 Watts) - 775,000 p/sec \n
GeForce 840M (790 GFLOPS, 30 Watts) - 1,080,000 p/sec
GeForce 960M (1403 GFLOPS, 65 Watts) - 3,570,000 p/sec
