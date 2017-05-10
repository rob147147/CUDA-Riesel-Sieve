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

cudaError_t addWithCuda(int *NOut, unsigned long long *KernelP, unsigned int size, const int blocks, const int threads, int *hashKeys, int *hashValues);


typedef struct _int128_t {
	unsigned long long int d1;
	unsigned long long int d0;
}int128_t;

//typedef struct _hashVec {
//	unsigned long long key;
//	int value;
//};

int *dev_a = 0; //NOut
unsigned long long *dev_b = 0; //KernelP
int *dev_c = 0; //kns
int *dev_e = 0; //Base
int *dev_f = 0; //counterIn
int *dev_g = 0; //HashTable Keys
int *dev_h = 0; //HashTable Values
cudaError_t cudaStatus;

__device__ int128_t div_128_64(unsigned long long int b) {
	/* Divides 2^128 by b, for 64-bit integer b, giving the quotient as the result. */
	int128_t quotient;

	unsigned long long int upper = 2;

	for (int i = 0; i<64; i++) {
		quotient.d1 = quotient.d1 << 1;
		if (b <= upper) {
			upper = upper - b;
			quotient.d1++;
		}
		upper = upper << 1;
	}

	for (int i = 0; i<64; i++) {
		quotient.d0 = quotient.d0 << 1;
		if (b <= upper) {
			upper = upper - b;
			quotient.d0++;
		}
		upper = upper << 1;
	}

	return quotient;
}

__device__ unsigned long long int barrett(int128_t m, unsigned long long int a, unsigned long long int b, unsigned long long int prime) {
	//We currently do 6*64-bit multiplies in here. We can probably reduce this to increase speed. 
	int128_t ab;
	//Calculate q = m.a.b/2^128. I.e. we only need the top 64 bits of m.a.b as m.a.b is at most 192 bits
	ab.d0 = a*b;
	ab.d1 = __umul64hi(a, b);

	//First two terms here are 0 if a and b are 32 bits or less as ab.d1=0
	unsigned long long int q = (ab.d1*m.d1) + __umul64hi(ab.d1, m.d0) + __umul64hi(ab.d0, m.d1);

	unsigned long long int r;
	//Calculate r = (a.b)-(q.n). This must be less than n so we only need the low 64 bits of (a.b) and (q.n)
	r = ab.d0 - (q*prime);

	if (r>prime) {
		r = r - prime;
	}

	return r;
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
			if ((r % 2) == 0) {
				r = r >> 1;
			}
			else {
				r = (r + b) >> 1;
			}
		}
		else {
			if ((v & 1) == 0) {
				v = v >> 1;
				if ((s % 2) == 0) {
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



__global__ void addKernel(int *NOut, unsigned long long *KernelP, int *kns, int *Base, int *counterIn)
{

    int i = threadIdx.x;
	int block = blockIdx.x;
	//Change this to somthing sensible in the future
	unsigned long long int b = KernelP[(128*block) + i];
	int KernelBase = *Base;
	//printf("%llu\n", b);
	//printf("%d\n", KernelBase);
	int counter = *counterIn;
	//printf("%d\n", counter);

	//Calculate m = floor(2^128/b) using div_128_64. 
	int128_t m1;
	m1 = div_128_64(b);

	int m = 512;
	int shift = 9; //m=2^shift

	//For all j s.t 0<=j<m, calculate KernelBase^j and store
	unsigned long long int js[512]; //Would like to call this m but it won't compile as the memory requirements must be known in advance
	js[0] = 1;
	for (int j = 1; j<m; j++) {
		js[j] = barrett(m1, js[j - 1], KernelBase, b);
		//count++;
	}

	//Compute KernelBase^-m (mod b)
	unsigned long long int c1 = binExtEuclid(KernelBase, b); //This should be KernelBase^-1 (mod b)
	//Now repeatedly square it as m is a power of two
	//printf("%d\n", c1);
	for (int s = 0; s<shift; s++) {
		c1 = barrett(m1, c1, c1, b);
		//count++;
	}

	//Lets try changing this section - rather than looking at every possible match lets just look for the ones we're interested in
	//The structure of the candidate file is 0,k,n-values,0,k,n-values,...
	//counter is the length of this array, so just work through it

	long long output = -5;

	for (int k = 0; k<counter; k++) {
		if (kns[k] == 0) {
			//The next entry is a k-value
			k++;
			//printf("%d\n", k);
			//So work out beta from it
			int kval = kns[k];
			unsigned long long int beta = binExtEuclid(kns[k], b);
			//printf("%d\n", beta);
			//The next value is the first n-value for this k-value
			k++;
			bool first = true;
			int t = 0;
			for (int z = 0; z<counter; z++) {
				//printf("%d\n", z);
				//Work through the n-values until we come across a zero, which implies we are done for this k-value
				if (kns[k + z] == 0) {
					k = k + z - 1; //This sets us back one value, so when the next loop starts and adds 1 to the value of k then we'll be located at a zero
					break; // z=counter
				}
				//Otherwise this is an n-value and we need to check it
				//Work out tMin -> take the n-value and divide by m
				int n = kns[k + z];
				if (first) {
					int tMin = n >> shift;
					for (t = 0; t<tMin; t++) {
						beta = barrett(m1, beta, c1, b);
						//count++;
					}
					//printf("%d\n", beta);
					first = false;
					t = t*m;
				}
				//Check the difference between t*m and the n-value;
				int diff = n - t;
				while (diff>m) { //Changing this if to while seems to result in odd runtime when changing the input arraysize when it should have no effect!
					diff = diff - m;
					t = t + m;
					beta = barrett(m1, beta, c1, b);
					//count++;
				}
				//count++;
				//printf("kval = %d, nval = %d, beta = %llu, jsdiff = %llu\n", kval, n, beta, js[diff]);
				//printf("jsdiff = %d\n", js[diff]);
				if ((beta) == js[diff]) {
					printf("We've had a match. Output will be %llu | %d*%d^%d-1\n", b,kval,KernelBase,n);
					output = kval;
					output = output << 32;
					output = output + t + diff;
				}
			}
		}
	}

	NOut[i] = output; //This contains the k-value in the top 32 bits and the n-value in the low 32 bits


}

__global__ void addKernel1(int *NOut, unsigned long long *KernelP, int *ks, int *Base, int *counterIn, int *hashKeys, int *hashValues)
{
	clock_t beginfull = clock();
	clock_t begin = clock();
	int i = threadIdx.x;
	int block = blockIdx.x;
	int N = 128 * 128; //This is threads*blocks
	int mem = 4096;
	int memN = N * mem;
	int S = block * 128; //This is threads*this block ID
	//Change this to somthing sensible in the future
	unsigned long long int b = KernelP[S + i];
	//unsigned long long int b = 600000000000000;
	int KernelBase = *Base;
	//printf("%llu\n", b);
	//printf("%d\n", KernelBase);
	int counter = *counterIn;
	//printf("%d\n", counter);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (i == 0 & block == 0) {
		//printf("Time to complete variable setup was %f\n", time_spent);
	}

	begin = clock();
	//Calculate m = floor(2^128/b) using div_128_64. 
	int128_t m1;
	m1 = div_128_64(b);

	int m = 1024;
	int shift = 10; //m=2^shift

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (i == 0 & block == 0) {
		//printf("Time to do long division was %f\n", time_spent);
	}

	begin = clock();
	//Try to populate our new hash table array ------------------------------------------------------------
	int lookups = 0;
	int hash = 0;
	//hashTable[1].key = 1;
	hashKeys[N+S+i] = 1;
	hashValues[N+S+i] = 0;
	unsigned long long js = 1;
	for (int j = 1; j<m; j++) {
		js = barrett(m1, js, KernelBase, b);
		hash = js & 4095;
		int index = hash*N + S + i;

		//Basic linear probing
		for (int probe = 0; probe < m; probe++) {
			lookups++;
			index = index % memN;
				
			if ((hashKeys[index]) == 0) {
				hashKeys[index] = js;
				hashValues[index] = j;
				break;
			}
			index = index + N;
		}
	}
	if (i == 0 & block == 0) {
		//printf("Number of lookups while inserting into the hash table was %d\n", lookups);
	}

	//Finished calculating the hash table --------------------------------------------------------------------

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (i == 0 & block == 0) {
		//printf("Time calculating new hash table was %f\n", time_spent);
	}


//	begin = clock();
//
//	//For all j s.t 0<=j<m, calculate KernelBase^j and store
//	_hashVec  hashTable[4096]; //This is our hash table with 2048 entires. Store js[j] mod 2048
//
//	//Set all the hash table keys to 0. Doesn't matter if there are leftover values in there
//
//	for (int j = 0; j < 4096; j++) {
//		hashTable[j].key = 0;
//	}
//
//	end = clock();
//	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	if (i == 0 & block == 0) {
//		printf("Time setting hash table to 0's was %f\n", time_spent);
//	}
//
//	begin = clock();
////Start calculating the hash table ------------------------------------------------------------------------
//
//	int lookups = 0;
//	int hash = 0;
//	hashTable[1].key = 1;
//	hashTable[1].value = 0;
//	unsigned long long js = 1;
//	for (int j = 1; j<m; j++) {
//		js = barrett(m1, js, KernelBase, b);
//		hash = js & 4095;
//		//printf("%llu is %d mod 2048. ", js[j], hash);
//		lookups++;
//		if ((hashTable[hash].key) == 0) {
//			hashTable[hash].key = js;
//			hashTable[hash].value = j;
//			if (i == 122 & block == 24) {
//				//printf("%d, %llu, %d\n", j, hashTable[hash].key, hashTable[hash].value);
//			}
//		}
//		else {
//			//Basic linear probing
//			for (int probe = 1; probe < m; probe++) {
//				lookups++;
//				if ((hashTable[(hash+probe)%4096].key) == 0) {
//					hashTable[(hash + probe) % 4096].key = js;
//					hashTable[(hash + probe) % 4096].value = j;
//					break;
//				}
//			}
//		}
//	}
//	if (i == 0 & block == 0) {
//		printf("Number of lookups while inserting into the hash table was %d\n", lookups);
//	}
//
////Finished calculating the hash table --------------------------------------------------------------------
//
//	end = clock();
//	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	if (i == 0 & block == 0) {
//		printf("Time calculating hash table was %f\n", time_spent);
//	}


	begin = clock();
	//Compute KernelBase^-m (mod b)
	unsigned long long int c1 = binExtEuclid(KernelBase, b); 
	//This should be KernelBase^-1 (mod b)
	//Now repeatedly square it as m is a power of two
															 
	for (int t = 0; t<shift; t++) {
		c1 = barrett(m1, c1, c1, b);
	}

	long long output = -5;
	int NMin = 20;
	int NMax = 1000;
	int tMin = NMin >> shift;
	lookups = 0;

	for (int k = 0; k<counter; k++) {
		//So work out beta from it
		unsigned long long beta = binExtEuclid(ks[k], b);
		
		for (int t = 0; t<tMin; t++) {
			beta = barrett(m1, beta, c1, b);
		}

		for (int t = tMin; t<70; t++) {
			//Check if beta is in js
			int hash = beta & 4095;
			int index = hash*N + S + i;
			index = index % memN;
			//If hashTable[hash] is empty then beta is not in js. If hashTable[hash]!=beta then use linear probing
			//if (hashTable[hash].key == 0) {
			//	//Beta is not here
			//}
			//else if ((hashTable[hash].key) == beta) {
			//	lookups++;
			//	//We've found beta
			//	//We've had a match
			//	output = (t*m + (hashTable[hash].value));
			//	//printf("Match via hash in Thread %d, Block %d. t=%d, w=%d, hash=%d, beta=%llu. Output will be %llu | %d*%d^%d-1\n", i, block, t, hashTable[hash].value, hash, beta , b, ks[k], KernelBase, output);
			//	break;
			//}
			//else {
			//	//Its possible beta is here, use linear probing to check
			//	for (int probe = 1; probe < 1024; probe++) {
			//		lookups++;
			//		if ((hashTable[(hash + probe) % 4096].key) == 0) {
			//			//Beta is not here
			//			break;
			//		}
			//		else if ((hashTable[(hash + probe) % 4096].key) == beta) {
			//			lookups++;
			//			//We've found beta
			//			//We've had a match
			//			output = (t*m + (hashTable[(hash + probe) % 4096].value));
			//			//printf("Match via hash with probing in Thread %d, Block %d. t=%d, w=%d, hash=%d, probe=%d beta=%llu. Output will be %llu | %d*%d^%d-1\n", i, block, t, hashTable[(hash + probe) % 2048].value, hash, probe, beta, b, ks[k], KernelBase, output);
			//			break;
			//		}
			//	}
			//}

			//Its possible beta is here, use linear probing to check
			for (int probe = 0; probe < 1024; probe++) {
				lookups++;
				if ((hashKeys[index]) == 0) {
					//Beta is not here
					break;
				}
				else if ((hashKeys[index]) == beta) {
					lookups++;
					//We've found beta
					//We've had a match
					output = (t*m + (hashValues[index]));
					//printf("Match via hash with probing in Thread %d, Block %d. t=%d, w=%d, hash=%d, probe=%d beta=%llu. Output will be %llu | %d*%d^%d-1\n", i, block, t, hashTable[(hash + probe) % 2048].value, hash, probe, beta, b, ks[k], KernelBase, output);
					break;
				}
				index = index + N;
				index = index % memN;
			}
			
			beta = barrett(m1, beta, c1, b);
		}

		if (output < NMin) {
			output=-3;
		}
		else if (output > NMax) {
			output = -4;
		}
		else {
			//printf("Output will be %llu | %d*%d^%d-1\n", b, ks[k], KernelBase, output);

		}
	}
	if (i == 0 & block == 0) {
		//printf("Number of lookups against hash table was %d\n", lookups);
	}

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (i == 0 & block == 0) {
		//printf("Time to complete BSGS step was %f\n", time_spent);
	}

	NOut[S + i] = output; //This contains the k-value in the top 32 bits and the n-value in the low 32 bits

	end = clock();
	time_spent = (double)(end - beginfull) / CLOCKS_PER_SEC;
	if (i == 0 & block == 0) {
		//printf("Full kernel execute time was %f\n", time_spent);
	}
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

	const int blocks = 128; 
	const int threads = 128; //These must multiply to around 65536. Larger and CUDA times out
    const int arraySize = blocks*threads;
	const int testArraySize = arraySize * 16;
	unsigned long long *KernelP = (unsigned long long *)malloc(arraySize*sizeof(unsigned long long));
    //unsigned long long KernelP[arraySize] = { 0 };
	int *NOut = (int *)malloc(arraySize*sizeof(int));
    //int NOut[arraySize] = { 0 };
	int *hashKeys = (int *)malloc(arraySize * 4096 * sizeof(int));
	memset(hashKeys, 0, arraySize * 4096 * sizeof(int));
	int *hashValues = (int *)malloc(arraySize * 4096 * sizeof(int));
	memset(hashValues, 0, arraySize * 4096 * sizeof(int));

	//Low should be greater than the primes we use below. 
	unsigned long long low = 60000000000;
	unsigned long long high = 60002000000;

	unsigned long long startLow = low; //Don't touch this. Used for timing purposes

	//Use the idea of a segmented sieve. Generate a list of small primes first
	//Could use the first 1024 primes as a starter. 8161 is the 1024th prime
	//Currently using the first 70 primes aas a starter.
	clock_t begin = clock();
	int smallPrimes = 350;
	int primeCount = 70;
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

	cudaStatus = cudaMalloc((void**)&dev_g, arraySize * 4096 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_h, arraySize * 4096 * sizeof(int));
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

	cudaStatus = cudaMemcpy(dev_g, hashKeys, arraySize * 4096 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_h, hashValues, arraySize * 4096 * sizeof(int), cudaMemcpyHostToDevice);
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
		cudaError_t cudaStatus = addWithCuda(NOut, KernelP, arraySize, blocks, threads, hashKeys, hashValues);
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
cudaError_t addWithCuda(int *NOut, unsigned long long *KernelP, unsigned int size, const int blocks, const int threads, int *hashKeys, int *hashValues)
{

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_b, KernelP, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed!");
    }

	cudaStatus = cudaMemset(dev_g, 0, size * 4096 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
	}


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
