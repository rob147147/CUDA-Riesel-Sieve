This is a sieve for eliminating non-primes of the form k*b^n-1, typically searched by the Conjectures 'R Us (CRUS) project over at mersenneforum.org.

<b>There is an executable available for download for Linux and Win64 (outdated), use the -h option to see the valid command line flags.</b>

The CPU is responsible for reading in ABCD files and generating arrays of prime numbers to send to the GPU.<br />
The GPU takes an array of prime numbers which may be prime factors of numbers of the form k*b^n-1. The GPU is also given a list of k-values and the n-min and n-max value from the ABCD file. 

The core algorithm run on the GPU is the Baby Steps Giants Steps (BSGS) algorithm for solving the discrete logarithm. 

<br />
This is very much a work in progress but these are some current performance numbers from CUDA enabled NVidia GPUs to which I have access.<br />
The test file is R745.ABCD which contains 22 k-values, with n-min = 180,000 and n-max = 250,000 giving an n-range of 70,000.<br />
<br />
CPU (1 core of i5-4440 @ 3.1Ghz, using SR2Sieve) - 6,000,000 p/sec<br />

<br />
Latest update as of 17/05/23:<br/>
<b>Nvidia GeForce RTX 2080 Ti (11750 GFLOPS, 11GB RAM, 5632kb L2 Cache, 250W) - 125,000,000 p/sec (with params -b 5 -m 4 -s 256 -Q 18) using the Linux executable</b> <br />
This appears to be GPU performance limited (CPU usage ~35%), and runs about 2-3 times quicker than srsieve2cl on the same GPU.  <br/><br/>
<b>Nvidia A100 80GB PCIe (19500 GFLOPS, 80GB RAM HBM2, 40960kb L2 Cache, 250W) - 360,000,000 p/sec (with params -b 5 -m 8 -s 192 -Q 18) using Linux executable v0.24.6</b> <br />
Changing the argument 's' makes the code vary between CPU and GPU bound, runs 2-3 times quicker than srsieve2cl on the same GPU.  <br/><br/>

<br/>
Old speed info for other GPUs: <br/>
Nvidia GeForce GTX 1060 (3855 GFLOPS, 6GB RAM, 1536kb L2 cache, 120 Watts) - ~23,000,000 p/sec @90% TDP ~108W (wth params -b 7 -m 5 -s 256 -Q 18) <br /> 
Nvidia MX150 (1177 GFLOPS, 2GB RAM, 512kb L2 cache, 25 Watts) - 6,500,000 p/sec (with params -b 5 -m 5)<br />
Nvidia GeForce RTX 2070 Max-Q (5460 GFLOPS, 8GB RAM, 4096kb L2 Cache, 80 Watts) - 54,800,000 p/sec (with params -b 9 -m 4)<br />
Nvidia GeForce RTX 2080 (8920 GFLOPS, 8GB RAM, 4096kb L2 Cache, 215W) - 80,000,000 p/sec (with params -b 7 -m 4 -s 256 -Q 18) <br />
Nvidia GeForce RTX 2080 Ti (11750 GFLOPS, 11GB RAM, 5632kb L2 Cache, 250W) - 123,000,000 p/sec (with params -b 8 -m 4 -s 256 -Q 18) <br />
<br /> <br />


