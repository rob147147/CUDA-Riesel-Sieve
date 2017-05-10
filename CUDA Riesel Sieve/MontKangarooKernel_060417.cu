//Complete - Read in an ABCD file
//Complete - CUDA code with correct outputs

//TODO - Fix inputs to the CUDA function call to be an actual ABCD file and list of primes
//TODO - Function to create list of primes


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>

using namespace std;

cudaError_t addWithCuda(int *NOut, unsigned long long *KernelP, unsigned int size, const int blocks, const int threads, unsigned long long *hashKeys, int *hashValues, int hashTableElements);


int *dev_a = 0; //NOut
unsigned long long *dev_b = 0; //KernelP
int *dev_c = 0; //kns
int *dev_e = 0; //Base
int *dev_f = 0; //counterIn
unsigned long long *dev_g = 0; //HashTable Keys
int *dev_h = 0; //HashTable Values
cudaError_t cudaStatus;


__device__ unsigned long long xbinGCD(unsigned long long a, unsigned long long b)
{
	unsigned long long alpha, beta, u, v;
	u = 1; v = 0;
	alpha = a; beta = b; // Note that alpha is
						 // even and beta is odd.
						 /* The invariant maintained from here on is:
						 2a = u*2*alpha - v*beta. */
	while (a > 0) {
		a = a >> 1;
		if ((u & 1) == 0) { // Delete a common
			u = u >> 1; v = v >> 1; // factor of 2 in
		} // u and v.
		else {
			/* We want to set u = (u + beta) >> 1, but
			that can overflow, so we use Dietz's method. */
			u = ((u ^ beta) >> 1) + (u & beta);
			v = (v >> 1) + alpha;
		}
	}
	//*pu = u;
	//*pv = v;
	return v;
}

__device__ unsigned long long modul64(unsigned long long x, unsigned long long y, unsigned long long z) {
	/* Divides (x || y) by z, for 64-bit integers x, y,
	and z, giving the remainder (modulus) as the result.
	Must have x < z (to get a 64-bit result). This is
	checked for. */
	long long i, t;
	if (x >= z) {
		printf("Bad call to modul64, must have x < z.");
	}
	for (i = 1; i <= 64; i++) { // Do 64 times.
		t = (long long)x >> 63; // All 1's if x(63) = 1.
		x = (x << 1) | (y >> 63); // Shift x || y left
		y = y << 1; // one bit.
		if ((x | t) >= z) {
			x = x - z;
			y = y + 1;
		}
	}
	return x; // Quotient is y.
}

__device__ __forceinline__ unsigned long long montmul(unsigned long long abar, unsigned long long bbar, unsigned long long m, unsigned long long mprime) {
	unsigned long long thi, tlo, tm, tmmhi, tmmlo;
	unsigned long long uhi, ulo;
	unsigned int ov;
	//mulul64(abar, bbar, &thi, &tlo); // t = abar*bbar.
	thi = __umul64hi(abar, bbar);
	tlo = abar*bbar;
	/* Now compute u = (t + ((t*mprime) & mask)*m) >> 64.
	The mask is fixed at 2**64-1. Because it is a 64-bit
	quantity, it suffices to compute the low-order 64
	bits of t*mprime, which means we can ignore thi. */
	tm = tlo*mprime;
	//mulul64(tm, m, &tmmhi, &tmmlo); // tmm = tm*m.
	tmmhi = __umul64hi(tm, m);
	tmmlo = tm*m;
	
	//Replace this with ptx
	//ulo = tlo + tmmlo; // Add t to tmm
	//uhi = thi + tmmhi; // (128-bit add).
	//if (ulo < tlo) uhi = uhi + 1; // Allow for a carry.
	// The above addition can overflow. Detect that here.
	//ov = (uhi < thi) | ((uhi == thi) & (ulo < tlo));

	asm("add.cc.u64 %0, %3, %4;\n\t" //Add tlo and tmmlo and set carry out. 
		"addc.cc.u64 %1, %5, %6;\n\t" //Add thi and tmmhi, use the previous carry and set carry out.
		"addc.u32 %2, 0, 0;" //This sets ov to 1 if the previous addition overflowed.
		: "=l"(ulo), "=l"(uhi) "=r"(ov) : "l"(tlo), "l"(tmmlo), "l"(thi), "l"(tmmhi)
		);

//	asm("add.cc.u64 %0, %0, %3;\n\t" //Add tlo and tmmlo and set carry out. 
//		"addc.cc.u64 %1, %1, %4;\n\t" //Add thi and tmmhi, use the previous carry and set carry out.
//		"addc.u32 %2, 0, 0;" //This sets ov to 1 if the previous addition overflowed.
//		: "=l"(tlo), "=l"(thi) "=r"(ov) : "l"(tmmlo), "l"(tmmhi)
//		);

//	if (ov > 0 || thi >= m) // If u >= m,
//		thi = thi - m; // subtract m from u.
//	return thi;



	ulo = uhi; // Shift u right
	if (ov > 0 || ulo >= m) // If u >= m,
		ulo = ulo - m; // subtract m from u.
	return ulo;
}

__device__ long long binExtEuclid(long long a, long long b) {
	long long u = b;
	long long v = a;
	long long r = 0;
	long long s = 1;
	long long x = a;
	while (v>0) {
		if ((u & 1) == 0) {
			u = u >> 1;
			if ((r & 1) == 0) {
				r = r >> 1;
			}
			else {
				r = (r + b) >> 1;
			}
		}
		else {
			if ((v & 1) == 0) {
				v = v >> 1;
				if ((s & 1) == 0) {
					s = s >> 1;
				}
				else {
					s = (s + b) >> 1;
				}
			}
			else {
				x = u - v;
				if (x>0) {
					u = x;
					r = r - s;
					if (r<0) {
						r = r + b;
					}
				}
				else {
					v = x * -1;
					s = s - r;
					if (s<0) {
						s = s + b;
					}
				}
			}
		}
	}
	if (r >= b) {
		r = r - b;
	}
	if (r<0) {
		r = r + b;
	}
	return(r);
}


__global__ void addKernel1(int *NOut, unsigned long long *KernelP, int *ks, int *Base, int *counterIn, unsigned long long *hashKeys, int *hashValues)
{
	
	int i = threadIdx.x;
	int block = blockIdx.x;
	int S = block * blockDim.x; //This is this block ID*threads in a block
	unsigned long long b = KernelP[S + i];
	unsigned long long bprime = xbinGCD(9223372036854775808, b);
	int KernelBase = *Base;
	int counter = *counterIn;
	int outputBase = KernelBase;

	int loop = 500; //This should be roughly sqrt(NMax - NMin)

	long output = -5;
	int NMax = 500000;
	int NMin = 250000;
	int d = NMax;

	//Move b1 to montgomery space
	unsigned long long b1 = modul64(KernelBase, 0, b);
	//These are all montgomery mults
	unsigned long long b2 = montmul(b1, b1, b, bprime);
	unsigned long long b3 = montmul(b2, b2, b, bprime);
	unsigned long long b4 = montmul(b3, b3, b, bprime);
	unsigned long long b5 = montmul(b4, b4, b, bprime);
	unsigned long long b6 = montmul(b5, b5, b, bprime);
	unsigned long long b7 = montmul(b6, b6, b, bprime);
	unsigned long long b8 = montmul(b7, b7, b, bprime);
	unsigned long long b9 = montmul(b8, b8, b, bprime);
	unsigned long long b10 = montmul(b9, b9, b, bprime);
	unsigned long long b11 = montmul(b10, b10, b, bprime);
	unsigned long long b12 = montmul(b11, b11, b, bprime);
	unsigned long long b13 = montmul(b12, b12, b, bprime);
	//ulong b14 = montmul(b13,b13,b,rInvMdash);
	//ulong b15 = montmul(b14,b14,b,rInvMdash);
	//ulong b16 = montmul(b15,b15,b,rInvMdash);

	//Move the x0 starting point to montgomery space
	unsigned long long x0 = modul64(1, 0, b);
	int tempNMax = NMax;

	unsigned long long bInc = b1;
	int k = 32 - __clz(tempNMax);
	for (int i = 0; i<k; i++) {
		if ((tempNMax & 1) == 1) {
			x0 = montmul(x0, bInc, b, bprime);
		}
		tempNMax >>= 1;
		bInc = montmul(bInc, bInc, b, bprime);
	}

	int j = 0;
	int j1 = 0;
	unsigned long long bsnew = 0;
	int distances[4] = { 1,32,256,512 }; //The mean of these should be roughly equal to loops/2
	unsigned long long bs[4] = { b1, b3, b9, b10};

	for (int i = 0; i<loop; i++) {
		j = ((x0)& 3);
		j1 = j & 1;
		bsnew = j1 == 0 ? j == 0 ? b1 : b9 : j == 1 ? b6 : b10;
		//d = d + (1 << j << j);
		x0 = montmul(x0, bsnew, b, bprime);
		//j = (x0) & 3;
		d = d + distances[j];
		//x0 = montmul(x0, bs[j], b, bprime);
		//printf("d is up to %d\n", d);
	}

	//The loop above is only run once as it doesn't depend on c1, and the loop below is run for each c1 (each k value)

	int permD = d;
	unsigned long long c1 = 0;

	for (int c = 0; c<counter; c++) {
		d = permD;
		bool xor = 1;
		output = -5;
		c1 = binExtEuclid(ks[c], b);
		//Move this to montgomery space
		c1 = modul64(c1, 0, b);
		while (xor) {
			j = ((c1)& 3);
			j1 = j & 1;
			bsnew = j1 == 0 ? j == 0 ? b1 : b9 : j == 1 ? b6 : b10;
			//d = d - (1 << j << j);
			c1 = montmul(c1, bsnew, b, bprime);
			//j = (x0) & 3;
			d = d - distances[j];
			//c1 = montmul(c1, bs[j], b, bprime);
			//printf("d is down to $d\n", d);

			xor = (c1 != x0);
			output = d;
			if (d<NMin) {
				xor = 0;
			}

		}
		if (output < NMin) {
			output = -3;
		}
		else if (output > NMax) {
			output = -4;
		}
		else {
			printf("Output will be %llu | %d*%d^%d-1\n", b, ks[c], outputBase, output);
		}

	}

	NOut[S + i] = output;
}



int main()
{

	//Read in an ABCD file and parse ----------------------------------------------------------------------------
	string line;
	int total = 0;
	//string abcdFile = "C:\\Users\\Rob\\Documents\\Visual Studio 2015\\Projects\\CPU Sieve\\sr_108.abcd";
	//string abcdFile = "C:\\Users\\Rob\\Desktop\\TestSieve\\sr_745.abcd";
	string abcdFile = "sr_745.abcd";

	//First pass through the ABCD file to find the number of k's and max number of n's
	int count1 = 0; //Number of k's
	int count3 = 0; //Total number of lines
	ifstream myfile(abcdFile);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			count3++;

			string::size_type n = line.find(" ");
			string token = line.substr(0,n);
			//cout << token << endl;

			//If tokens[0] == "ABCD" then this defines a new k, otherwise it is a number
			if (token.compare("ABCD") == 0) {
				count1++;
				//cout << "We're here!" << endl;
			}
		}
		myfile.close();
	}

	else cout << "Unable to open file first time" << endl;

	//Second pass through the ABCD file to write the values into the matrix
	//Store the k and n values in this array
	//boost::numeric::ublas::matrix<int> kns(count1, max);
	//std::list<int> kns;
	count3 = count3 + (2 * count1);
	int *kns = (int *)malloc(count3*sizeof(int));
	int *ks = (int *)malloc(count1*sizeof(int));

	//Reset the counts
	count1 = 0;
	count3 = 0;
	int base = 0;
	ifstream myfile2(abcdFile);
	if (myfile2.is_open())
	{
		while (getline(myfile2, line))
		{
			//Tokenise the string - if the first element of the string is "ABCD" then this is a new k-value
			string::size_type n = line.find(" ");
			string token = line.substr(0, n);
			//If tokens[0] == "ABCD" then this defines a new k, otherwise it is a number
			if (token.compare("ABCD") == 0) {
				cout << "We've found ABCD. Get the k-value" << endl;
				//Insert a 0 into kns before we insert the k-value
				kns[count3] = 0;
				count3++;
				//Get the k value
				token = line.substr(n+1);
				//cout << token << endl;
				n = token.find("*");
				string tok = token.substr(0, n);
				//cout << tok << endl;
				
				int kval = stoi(tok);
				kns[count3] = kval;
				count3++;
				ks[count1] = kval;
				count1++;
				//Get the base
				if (base == 0) {
					token = token.substr(n+1);
					n = token.find("^");
					string b = token.substr(0, n);
					//cout << b << endl;
					base = stoi(b);
					cout << "The base is " << base << endl;
				}
				//Get the starting n-value - remove the square brakets
				n = token.find("[");
				token = token.substr(n + 1);
				n = token.find("]");
				token = token.substr(0, n);
				//cout << token << endl;
				total = stoi(token);
				kns[count3] = total;
				count3++;
				cout << "This is a new k-value with value " << kval << " and initial n-value " << total << endl;
			}
			else {
				//This is a number, n-value offset
				//cout << token << endl;
				int offset = stoi(token);
				total = total + offset;
				kns[count3] = total;
				count3++;
				//cout << count3 << endl;
			}

		}

		myfile2.close();
	}

	else cout << "Unable to open file second time" << endl;

	//End of reading ABCD file ----------------------------------------------------------------------------------
	cout << "End of reading ABCD file" << endl;


	//Generate Primes -------------------------------------------------------------------------------------------

	const int blocks = 256; 
	const int threads = 128; //These must multiply to around 65536. Larger and CUDA times out
    const int arraySize = blocks*threads;
	const int testArraySize = arraySize * 24;
	const int hashTableElements = 512;
	unsigned long long *KernelP = (unsigned long long *)malloc(arraySize*sizeof(unsigned long long));
    //unsigned long long KernelP[arraySize] = { 0 };
	int *NOut = (int *)malloc(arraySize*sizeof(int));
    //int NOut[arraySize] = { 0 };
	unsigned long long *hashKeys = (unsigned long long *)malloc(arraySize * hashTableElements * 4 * sizeof(unsigned long long));
	memset(hashKeys, 0, arraySize * hashTableElements * 4 * sizeof(unsigned long long));
	int *hashValues = (int *)malloc(arraySize * hashTableElements * 4 * sizeof(int));
	memset(hashValues, 0, arraySize * hashTableElements * 4 * sizeof(int));

	//Low should be greater than the primes we use below. 
	//unsigned long long low = 6000000000;
	//unsigned long long high = 6003000000;

	unsigned long long low = 1000067000000;
	unsigned long long high = 1000070000000;

	//unsigned long long low = 600000;
	//unsigned long long high = 7000000;

	unsigned long long startLow = low; //Don't touch this. Used for timing purposes

	//Use the idea of a segmented sieve. Generate a list of small primes first
	//Could use the first 1024 primes as a starter. 8161 is the 1024th prime
	//Currently using the first 70 primes aas a starter.
	clock_t begin = clock();
	int smallPrimes = 8162;
	int primeCount = 1024;
	int s = 0;
	bool *primes = (bool *)malloc(smallPrimes*sizeof(bool));
	unsigned int *smallP = (unsigned int *)malloc(primeCount*sizeof(unsigned int));
	memset(primes, true, smallPrimes*sizeof(bool));

	int sq = smallPrimes*smallPrimes;

	for (int p = 2; p*p < sq; p++) {
		if (primes[p] == true) {
			smallP[s] = p;
			//cout << smallP[s] << endl;
			s++;
			for (int i = p*2; i < smallPrimes; i += p) {
				primes[i] = false;
			}
		}
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Time generating small primes " << time_spent << "s" << endl;

	//Print the small primes as a check
	for (int p = 0; p < primeCount; p++) {
		//cout << smallP[p] << endl;
	}

	//Find the minimum number in [low...high] that is a multiple of primes[i]
	
	bool *mark = (bool *)malloc(testArraySize*sizeof(bool));



	//Try setting up the GPU just once

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (one output, seven input). 
	//Give all vectors same size for now, we can change this afterwards

	cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_c, count1 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_e, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_f, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_g, arraySize * hashTableElements * 4 * sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_h, arraySize * hashTableElements * 4 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_c, ks, count1 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_e, &base, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_f, &count1, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_g, hashKeys, arraySize * hashTableElements * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_h, hashValues, arraySize * hashTableElements * 4 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
		goto Error;
	}

	int kernelCount = 0;
	clock_t loopTime = clock();
	//From here we need to loop to keep the GPU busy. 
	while (low < high) {
		kernelCount++;
		cout << "Executing kernel number " << kernelCount << endl;
		cout << "Low is now set to " << low << endl;

		begin = clock();
		memset(mark, true, testArraySize*sizeof(bool));

		for (int i = 0; i < primeCount; i++) {
			unsigned int smallPrime = smallP[i];
			for (int j = 0; j < testArraySize; j++) {
				//if (mark[j] == true && (((low + j) % smallP[i]) == 0)) {
				if (((low + j) % smallPrime) == 0) {
					//So if low + offset can be divided by i we've found the first value divisible by i. Now mark off all i multiples
					for (int k = j; k < testArraySize; k += smallPrime) {
						mark[k] = false;
					}
					break;
				}
			}
		}
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "Time marking the prime array " << time_spent << "s" << endl;


		// Numbers which are not marked as false are prime
		begin = clock();
		int countPrimes = 0;
		for (unsigned long long i = low; i < low + (testArraySize); i++) {
			if (mark[i - low] == true) {
				KernelP[countPrimes] = i;
				countPrimes++;
				if (countPrimes == arraySize) {
					cout << "We got as far as " << i << " out of " << low + (testArraySize) << endl;
					break;
				}
				//cout << i << endl;
			}
		}
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "Time generating kernel primes " << time_spent << "s" << endl;

		unsigned long long minPrime = KernelP[0];
		unsigned long long maxPrime = KernelP[arraySize - 1];
		unsigned long long progress = maxPrime - minPrime;

		cout << "Min Prime = " << minPrime << ". Max Prime = " << maxPrime << ". Progress = " << progress << endl;
		cout << "Array Size = " << arraySize << endl;

		//End of Generating Primes ----------------------------------------------------------------------------------

		begin = clock();
		cout << "Try to launch the CUDA kernel" << endl;
		// Add vectors in parallel.
		//This uses the full ABCD file, but runs very slowly when file is big
		//cudaError_t cudaStatus = addWithCuda(NOut, KernelP, kns, &base, &count3, arraySize, count3, blocks, threads);
		//This is datless - remember to change to addkernel1
		cudaError_t cudaStatus = addWithCuda(NOut, KernelP, arraySize, blocks, threads, hashKeys, hashValues, hashTableElements);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "Time to execute kernel (outside function) " << time_spent << "s" << endl;

		printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n\n",
			NOut[0], NOut[1], NOut[2], NOut[3], NOut[4], NOut[5], NOut[6], NOut[7], NOut[8], NOut[9]);

		//Set low equal to high and continue in the loop
		low = maxPrime;
	}

	clock_t loopEnd = clock();
	time_spent = (double)(loopEnd - loopTime) / CLOCKS_PER_SEC;
	cout << "Time taken " << time_spent << "s" << endl;
	cout << "Time per kernel " << time_spent / kernelCount << endl;
	cout << "Progress = " << KernelP[arraySize - 1] - startLow << " at " << (KernelP[arraySize - 1] - startLow) / time_spent << " p/sec" << endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_e);
	cudaFree(dev_f);
	cudaFree(dev_g);
	cudaFree(dev_h);
	return cudaStatus;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *NOut, unsigned long long *KernelP, unsigned int size, const int blocks, const int threads, unsigned long long *hashKeys, int *hashValues, int hashTableElements)
{

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_b, KernelP, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed!");
    }

	cudaStatus = cudaMemset(dev_g, 0, size * hashTableElements * 4 * sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
	}

	//cudaStatus = cudaMemset(dev_h, 0, size * hashTableElements * 4 * sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy input failed!");
	//}


	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
    // Launch a kernel on the GPU with one thread for each element.
	//cudaEventRecord(start);
	cudaStream_t stream0;
	//cudaStream_t stream1;
	cudaStreamCreate(&stream0);
	//cudaStreamCreate(&stream1);
    addKernel1<<<blocks,threads,0,stream0>>>(dev_a, dev_b, dev_c, dev_e, dev_f, dev_g, dev_h);
	//addKernel1<<<blocks,threads,0,stream1>>>(dev_a, dev_b, dev_c, dev_e, dev_f, dev_g, dev_h);
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("Time taken: %f ms \n", milliseconds);

    // Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //}
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(NOut, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy output failed!");
    }
    
    return cudaStatus;
}
