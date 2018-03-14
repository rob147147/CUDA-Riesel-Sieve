This is a sieve for eliminating non-primes of the form k*b^n-1, typically searched by the Conjectures R'Us (CRUS) project over at mersenneforum.org.

The CPU is responsible for reading in ABCD files and generating arrays of prime numbers to send to the GPU.<br />
The GPU takes an array of prime numbers which may be prime factors of numbers of the form k*b^n-1. The GPU is also given a list of k-values and the n-min and n-max value from the ABCD file. 

The core algorithm run on the GPU is the Baby Steps Giants Steps (BSGS) algorithm for solving the discrete logarithm. 

<br />
This is very much a work in progress but these are some current performance numbers from CUDA enabled NVidia GPUs to which I have access.<br />
The test file is R745.ABCD which contains 22 k-values, with n-min = 180,000 and n-max = 250,000 giving an n-range of 70,000.<br />
<br />
CPU (1 core of i5-4440 @ 3.1Ghz) - 6,000,000 p/sec<br />
GeForce GT 710 (366 GFLOPS, 1GB RAM, 512kb L2 cache, 19 Watts) - 845,000 p/sec (with options 4 16) <br />
GeForce 840M (790 GFLOPS, 2GB RAM, 1024kb L2 cache 30 Watts) - ?? p/sec<br />
GeForce 960M (1403 GFLOPS, 2GB RAM, 2048kb L2 cache, 65 Watts) - ~4,100,000 p/sec (with options 8 16) <br />
GeForce GTX 1060 (3855 GFLOPS, 6GB RAM, 1536kb L2 cache, 120 Watts) - ~8,400,000 p/sec @90% TDP ~110W (wth options 8 64) <br /> 