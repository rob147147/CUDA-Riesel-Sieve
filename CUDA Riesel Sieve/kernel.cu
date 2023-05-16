//TODO:
//Device selector in case of multiple CUDA enabled GPUs
//Check arguments are sensible/valid
//Check sierpinksi abcd files for correctness and send the right info for this to the gpu to avoid the manual switch
//Error checking, particularly around the reading of an ABCD file
//Make Error a function instead of using goto
//Release memory on completion
//Check to see if we have a memory leak somewhere as implied on NCC
//Check arguments won't request too much memory - catching potential CUDA memory allocation errors before they happen.
//Make the output based on a timer with an input argument for it

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <omp.h>
//#include <FileInput.h>
//#include "FileInput.h"

using namespace std;

void generateGPUPrimes(unsigned int *KernelP1, unsigned long long low, unsigned int *smallP, int testArraySize, int primeCount, int arraySize, unsigned long long *mark1, unsigned long long *mask, unsigned long long *mask1, unsigned long long *mask2, unsigned long long *mask3);
void writeFactorToFile(unsigned long long p, unsigned int k, unsigned int b, unsigned int n, char *factorFile);

#define PRINT true
#undef PRINT

int *dev_a = 0; //NOut
unsigned int *dev_b = 0; //KernelP
int *dev_c = 0; //kns
__constant__ int base[1]; //Base
__constant__ unsigned long long lowGPU[1]; //
int *dev_f; //counterIn
int *dev_g = 0; //HashTable Keys
int *dev_h = 0; //HashTableElements
int *dev_i = 0; //HashTableDensity
unsigned int *dev_j = 0;
int *dev_k = 0; //Q
__constant__ int tMin[1]; //tMin
__constant__ int tMax[1]; //tMax
int *dev_n = 0; //minSubs

cudaError_t cudaStatus;


__device__ __forceinline__ int legendre(unsigned int a, unsigned long long p) {
	//Work out the legendre symbol for (a/p)
	//This code is taken straight from the source code of SR2Sieve
	unsigned int x, y;
	//Odd sign is positive(sign&1==1), even sign is negative(sign&1==0)
	unsigned int sign = 1;
	for (y = a; y % 2 == 0; y /= 2) {
		if (p % 8 == 3 || p % 8 == 5) {
			sign++;
		}
	}
	if (p % 4 == 3 && y % 4 == 3) {
		sign++;
	}

	unsigned long long xtemp = p % y;

	for (x = int(xtemp); x>0; x %= y) {
		for (; x % 2 == 0; x /= 2) {
			if (y % 8 == 3 || y % 8 == 5) {
				sign++;
			}
		}
		//Swap x and y
		x = x ^ y;
		y = x ^ y;
		x = x ^ y;

		if (x % 4 == 3 && y % 4 == 3) {
			sign++;
		}
	}

	return sign & 1;
}


__device__  __forceinline__ void xbinGCDnew(unsigned long long beta, unsigned long long &v)
{
	unsigned long long alpha = 9223372036854775808;
	unsigned long long u = 1;
	// Note that alpha is even and beta is odd.
	// The invariant maintained from here on is: 2a = u*2*alpha - v*beta.

	#pragma unroll 1
	for (int i=0; i<64; i++) {
		v = v >> 1;
		if ((u & 1) == 0) {
			u = u >> 1; // Delete a common factor of 2 in u and v.
		}
		else {
			/* We want to set u = (u + beta) >> 1, but that can overflow, so we use Dietz's method. */
			u = ((u ^ beta) >> 1) + (u & beta);
			v += alpha; //v>>1 happens in both cases, this just also sets the highest bit to 1
		}
	}
}

void countKs(char* abcdFile, int& count1, int& count3) {
	string line;

	ifstream myfile(abcdFile);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			count3++;

			string::size_type n = line.find(" ");
			string token = line.substr(0, n);
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
}

int core(unsigned int k) {
	//Return the square free part of k
	//Basic method - if remainder after dividing by a prime is 0 twice, we can remove this from k
	//At the moment just remove a single 2^2, 3^2 and/or 5^2
	if (k % 2 == 0) {
		k = k / 2;
		if (k % 2 == 0) {
			k = k / 2;
		}
		else {
			k = k * 2;
		}
	}
	if (k % 3 == 0) {
		k = k / 3;
		if (k % 3 == 0) {
			k = k / 3;
		}
		else {
			k = k * 3;
		}
	}
	if (k % 5 == 0) {
		k = k / 5;
		if (k % 5 == 0) {
			k = k / 5;
		}
		else {
			k = k * 5;
		}
	}
	return k;
}


__device__ __forceinline__ unsigned long long modul64(unsigned long long x, unsigned long long y, unsigned long long z) {
	/* Divides (x || y) by z, for 64-bit integers x, y,
	and z, giving the remainder (modulus) as the result.
	Must have x < z (to get a 64-bit result). This is
	checked for. */

	//If we limit z to being less 2^63, then x will always have a 0 first bit (as x < z)
	//In which case t will always be 0, so is not needed. 
	//Even when we shift x left (double it) after we subtract z it will never have its first bit set. 

	//long long t;
	#ifdef PRINT
	if (x >= z) {
		printf("Bad call to modul64, must have x < z.");
	}
	#endif

#pragma unroll 1
	for (int i = 0; i < 64; i++) { // Do 64 times.
		//t = (long long)x >> 63; // All 1's if x(63) = 1?
		x = (x << 1) | (y >> 63); // Shift x || y left      <- Bitwise OR?
		y = y << 1; // one bit.
		//if ((x | t) >= z) {
		if (x >= z) {
			x = x - z;
			y = y + 1;
		}
	}
	return x; // Quotient is y.
}


__device__ __forceinline__ unsigned long long montmul(unsigned long long abar, unsigned long long bbar, unsigned int mlo, unsigned int mhi, unsigned int mprimelo, unsigned int mprimehi) {

	//Take the 64 bit inputs, but do all multiplies in 32 bit chunks
	unsigned int alo = (unsigned int)abar;
	unsigned int ahi = (unsigned int)(abar >> 32);
	unsigned int blo = (unsigned int)bbar;
	unsigned int bhi = (unsigned int)(bbar >> 32);

	unsigned int u0 = 0;
	unsigned int u1 = 0;
	unsigned int u2 = 0;
	unsigned int u3 = 0;


	//Maybe just calculate u0 and u1 to start with as we need them. Then deal with u2 and u3 entirely afterwards. Might be more efficient overall
	//modul64 limits us to primes less than 2^63, so we might as well use that fact. This limits abar, bbar and m to 2^63 too.
	//t = abar*bbar is now limited to 2^126
	//t*mprime&mask can still be (2^64)-1
	//tm*m is now limited to less than 2^127 (about 2^126.5 roughly)
	//The sum u is now less than 2^128-1 in all cases, so no overflow to worry about
	
	//We need to calculate all 128 bits of t = abar * bbar. Add straight to u
	//PTX Version 1
	asm("{.reg .u32 treg1;\n\t"              // temp reg t1
		".reg.u32 treg2; \n\t"              // temp reg t2
		"mul.hi.u32 %1, %4, %6;\n\t" //Bits 32-64 (u1)
		"mul.lo.u32 %2, %5, %7;\n\t" //Bits 65-96 (u2)
		"mul.lo.u32 %0, %4, %6; \n\t" //Lowest 32 bits of u (u0)
		"mad.lo.cc.u32 %1, %4, %7, %1;\n\t" //Add the crossproduct to u1. Set the carry out
		"madc.hi.cc.u32 %2, %4, %7, %2;\n\t" //Add the crossproduct to u2. Set the carry out, and use the carry in
		"madc.hi.u32 %3, %5, %7, %3;\n\t" //Bits 97-128 plus any carry out (u3)
		"mad.lo.cc.u32 %1, %5, %6, %1;\n\t" //Add the other crossproduct to u1. Set the carry out
		"madc.hi.cc.u32 %2, %5, %6, %2;\n\t" //Add the other crossproduct to u2. Set the carry out, and use the carry in
		"addc.u32 %3, %3, 0;\n\t" //Add the potential carry into u3

	//Now we calculate t*mprime & mask. I.e. the low 64 bits
		"mul.lo.u32 treg2, %0, %11;\n\t" //Add the lo part of the crossproduct to tm1. u0*mprimehi
		"mad.lo.u32 treg2, %1, %10, treg2;\n\t" //Add the lo part of the other crossproduct to tm1 u1*mprimelo
		"mul.lo.u32 treg1, %0, %10;\n\t" //tm0 = lo part of u0*mprimelo
		"mad.hi.u32 treg2, %0, %10, treg2;\n\t" //tm1

	//Multiple tm*m to get the 128 bit product, and add it to u
	//If we move these first four lines to the end of the block we can save the two overflow additions putting flags into u0
		//"mad.lo.cc.u32 %0, treg1, %8, %0;\n\t" //Add the lo part of tm.lo*m*lo to u0. Set the carry out
		"add.u32 %1, %1, 1;\n\t" //Don't bother with t0, it will always be 0 and generate a carry
		"mad.hi.cc.u32 %1, treg1, %8, %1;\n\t" //Add the hi part of tm.lo*m*lo to u1. Set the carry out, and use the carry in
		"madc.lo.cc.u32 %2, treg2, %9, %2;\n\t" //Add the lo part of tm.hi*m*hi to u2. Set the carry out, and use the carry in
		"madc.hi.cc.u32 %3, treg2, %9, %3;\n\t" //Add the hi part of tm.hi*m*hi to u3. Use the carry in, and set carry out for overflow detection
		//"addc.u32 %0, 0, 0;\n\t" //Put the overflow flag into u0
		"mad.lo.cc.u32 %1, treg1, %9, %1;\n\t" //Add the lo part of tm.lo*m*hi to u1. Set the carry out
		"madc.hi.cc.u32 %2, treg1, %9, %2;\n\t" //Add the hi part of tm.lo*m*hi to u2. Set the carry out, and use the carry in
		"addc.u32 %3, %3, 0;\n\t" //Add the potential carry into u3
		//"addc.u32 %0, %0, 0;\n\t" //Put the overflow flag into u0
		"mad.lo.cc.u32 %1, treg2, %8, %1;\n\t" //Add the lo part of tm.hi*m*lo to u1. Set the carry out
		"madc.hi.cc.u32 %2, treg2, %8, %2;\n\t" //Add the hi part of tm.hi*m*lo to u2. Set the carry out, and use the carry in
		"addc.u32 %3, %3, 0\n\t;}" //Add the potential carry into u3
		//"addc.u32 %0, %0, 0;}" //Put the overflow flag into u0

		: "+r"(u0), "+r"(u1), "+r"(u2), "+r"(u3) : "r"(alo), "r"(ahi), "r"(blo), "r"(bhi), "r"(mlo), "r"(mhi), "r"(mprimelo), "r"(mprimehi)
	);

	//Look at this, we don't need to save tm0 and tm1 as they are not used after the asm. 
	//if (mlo == ((unsigned int)102297149781479)) {
	//	printf("Inputs = %llu, %llu, %u, %u, %u, %u\n", abar,bbar,mlo,mhi,mprimelo,mprimehi);
	//	printf("u = %u %u %u %u\n", u3, u2, u1, u0);
	//}


	//We're only interested in u2 and u3
	unsigned long long u = u3;
	u = u << 32;
	u = u | u2;

	unsigned long long m = mhi;
	m = m << 32;
	m = m | mlo;

	//Confirmed that this piece of code is useful and cases do fall into here
	if (u >= m) {
		u = u - m;
	}
	
	//if (mlo == ((unsigned int)102297149781479)) {
	//	printf("Output = %llu\n", u);
	//}

	return u;

}


__global__ void addKernel1(int *NOut, unsigned int *KernelP, int *knmatrix, int *rowOffset, int *hashKeys, int *hashElements, int *hashDensity, unsigned int *bits, int *Q, int *minSubs)
{
	#ifdef PRINT
	clock_t beginfull = clock();
	clock_t begin = clock();

	int legtime = 0;
	clock_t beginLeg = clock();
	clock_t endLeg = clock();

	clock_t end = clock();
	int time_spent = 0;
	#endif

	//This deals with the hashTables
	const int m = *hashElements;
	const int mem = m * (*hashDensity); //This is hashTableElements*density, to keep the correct thread using correct hash table 

	//m=2^shift, calculate shift
	int shift = 31 - __clz(m);

	const int S = (blockIdx.x * blockDim.x) + threadIdx.x; //This is this block ID*threads in a block + threadID
 const int SFull = gridDim.x * blockDim.x;
	const int Sm = S * mem;
	const int Sints = Sm >> 5;

	bool printer = false;
	if (S == 0) {
		printer = true;
	}

	const unsigned long long b = KernelP[S] + lowGPU[0]; //Lowest prime sent to GPU + offsets. Saves memory as offsets are ints rather than longs
	unsigned long long oneMS = modul64(1, 0, b); 
	//const unsigned long long oneMS = modul64(18446744073709551614, 0, b);

	unsigned long long bprime = 0;

	#ifdef PRINT
	int montmuls = 0;
	int montmuls1 = 0;
	int modul = 0;
	int bitLookups = 0;
	int bitUpdates = 0;
	int inserts = 0;

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("KernelBase = %d\n", base[0]);
		printf("HashTableElements = %d. %d at 1/%d density.\n", mem, m, *hashDensity);
		//printf("Each Thread should use %d ints in its bit array.\n", ints);
		printf("Q = %d.\n", *Q);
		printf("Cycles to complete variable setup was %d\n", time_spent);
	}

	begin = clock();
	#endif

	xbinGCDnew(b, bprime);

	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		//Check GCD has worked
		//printf("2*inp*%llu - %llu*%llu = %llu\n", rInv, b, bprime, (2 * 9223372036854775808 * rInv - (b*bprime)));
		printf("Cycles to do xbinGCD was %d\n", time_spent);
	}
	#endif

	unsigned long long KernelBase = modul64(base[0], 0, b);
	unsigned long long newKB = oneMS;

	unsigned int mlo = (unsigned int)b;
	unsigned int mhi = (unsigned int)(b >> 32);
	unsigned int mprimelo = (unsigned int)bprime;
	unsigned int mprimehi = (unsigned int)(bprime >> 32);

	//We now deal with b^Q for subsequences. 
	for (int qq = 0; qq < *Q; qq++) {
		newKB = montmul(KernelBase, newKB, mlo, mhi, mprimelo, mprimehi);
	}

	unsigned long long js = oneMS;

	//Do a dry run through the baby steps to find the free positions in the hash table
	#ifdef PRINT
	begin = clock();
	#endif

	unsigned int hash = 0;

	for (int j = 0; j < m; j++) {
		hash = (int)js & (m - 1);
		bits[Sints + (hash >> 5)] |= (1 << (hash & 31));
		js = montmul(js, newKB, mlo, mhi, mprimelo, mprimehi);
	}

	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles doing dry run of new hash table was %d (%d inserts (baby steps) @ %d cycles average)\n", time_spent, m, time_spent / m);
	}
	#endif

	//Try to populate our new hash table array ------------------------------------------------------------
	//New method - pre-populate with non-collision elements. Backfill the spaces.
	js = oneMS;
	int lookups = 0;
	int store = 0;
	int key = 0;
	int pointer = 0;
	int firstFree = 0; //The first memory cell that doesn't head a linked list

	for (int j = 0; j < m; j++) {

		hash = (int)js & (m - 1);
		store = (js & 0xFFFF0000) + 0x00007FFF; //This blanks off the last 16 bits and adds a null pointer. This will contain our pointer

		key = hashKeys[(Sm + hash)];

		if (key != 0) {
			//You were a collision into this bucket. Find somewhere to live, and update the pointer
			while (((bits[Sints + (firstFree >> 5)] >> (firstFree & 31)) & 1) == 1) {
				firstFree++;
			}

			hashKeys[(Sm + firstFree)] = ((store & 0xFFFF0000) + ((key & 0x0000FFFF) | 0x00008000)); //Store this new data, with the pointer from the head. We're now 2nd in this linked list
			store = (key & 0xFFFF0000) + firstFree;
			firstFree++;

		}

		hashKeys[(Sm + hash)] = store; //Update the linked list head, either with new data and a null pointer, or an updated pointer

		js = montmul(js, newKB, mlo, mhi, mprimelo, mprimehi);

	}

	//Finished calculating the hash table --------------------------------------------------------------------


	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles calculating new hash table was %d (%d inserts (baby steps) @ %d cycles average)\n", time_spent, m, time_spent / m);
	}
	begin = clock();
	#endif

	//c1  should be KernelBase^Q^-1 (mod b) computed earlier
	//Now repeatedly square it as m is a power of two

	unsigned long long c1 = newKB;

	for (int t = 0; t < shift; t++) {
		c1 = montmul(c1, c1, mlo, mhi, mprimelo, mprimehi);
	}

	int output = -1;

	#ifdef PRINT
	if (printer) {
		printf("tMin = %d. tMax = %d\n", tMin[0], tMax[0]);
	}
	#endif

	lookups = 0;
	int countmuls = tMin[0];
	#ifdef PRINT
	int giant = 0;
	int maxProbe = 0;
	int keyHit = 0;
	#endif

	unsigned long long fixedBeta = 0;
	unsigned long long beta = oneMS;
	int prevRem = 0;

	for (int t = 0; t < tMin[0]; t++) {
		beta = montmul(beta, c1, mlo, mhi, mprimelo, mprimehi);
	}

	const int subseqCount = *minSubs;
	const int offset = *rowOffset;
	int leg1;
	int leg2fixed = legendre(base[0], b);
	int leg2;
	int leg = 1;
	int leg12;

	int probe = 0;
	int thisk = 0;
	int corek = 0;
	int lastk = 0;
	int remainder = 0;
	int hash_time_spent = 0;
	int loop_time_spent = 0;
	int wloop_time_spent = 0;
	int init_time_spent = 0;

	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles setting up the second half was %d\n", time_spent);
	}
	#endif

	//Work through the matrix of kn values
	for (int k = 0; k < subseqCount; k++) {
		#ifdef PRINT
		int beginInitLoop = clock();
		#endif	
		lastk = thisk;
		thisk = knmatrix[k * offset];
		remainder = knmatrix[(k * offset) + 1];
		if (thisk != lastk) {
			#ifdef PRINT
			beginLeg = clock();
			#endif	
			leg1 = legendre(thisk, b);
			leg2 = !(leg1^leg2fixed); //Rather than use k*base, use the multiplicative property of legendre to save any overflows
			leg12 = leg1 || leg2;
			#ifdef PRINT
			leg = leg + 1;
			endLeg = clock();
			time_spent = (endLeg - beginLeg);
			legtime += time_spent;
			#endif		
			fixedBeta = modul64(thisk, 0, b);
			//The single line of code below lets us sieve +1 instead of -1
			//fixedBeta = b - fixedBeta;
			fixedBeta = montmul(fixedBeta, beta, mlo, mhi, mprimelo, mprimehi); 
			prevRem = 0;
		}

		#ifdef PRINT
		int endInitLoop = clock();
		init_time_spent += (endInitLoop - beginInitLoop);
		#endif	

		if (leg12) {
			#ifdef PRINT
			int beginHash = clock();
			#endif	
			//remainder = knmatrix[(k * offset) + 1];
			if ((remainder % 2 == 0 && leg1) || (remainder % 2 == 1 && leg2)) {

				//We need to do something
				//unsigned long long sB = fixedBeta;
				for (int rem = prevRem; rem < remainder; rem++) {
					fixedBeta = montmul(fixedBeta, KernelBase, mlo, mhi, mprimelo, mprimehi);
				}
				prevRem = remainder;
				unsigned long long sB = fixedBeta;
				#ifdef PRINT
				int beginLoop = clock();
				#endif	
				//int off = 2;
				for (int t = tMin[0]; t < tMax[0]; t++) {
					sB = montmul(sB, c1, mlo, mhi, mprimelo, mprimehi);
					//This is wrong. The bits that we look at need to be based on t, as this is impacted by choice of m.
					//int candidates = knmatrix[(k* *rowOffset) + off];
					//off++;
					//if (candidates == 0) {
					//	continue;
					//}
					#ifdef PRINT
					giant++;
					probe = 0;
					#endif
					//int sBint = (int)sB; <- Look to see if storing this is useful. 

					//Check if beta is in js
					bool first = true;
					int sbInt = (int)sB;
					int sbHigh = sbInt & 0xFFFF0000;
					pointer = sbInt & (m - 1);

					#ifdef PRINT
					int beginWLoop = clock();
					#endif	

					while (true) {

						//This was quicker with the bit array in the past - it now appears to be faster without using the bit array
						key = hashKeys[(Sm + pointer)];
						lookups++;

						#ifdef PRINT
						probe++;
						if (probe > maxProbe) {
							maxProbe = probe;
						}
						#endif

						pointer = (key & 0x0000FFFF); //Remove the data, leave the pointer
						key = key & 0xFFFF0000; //Remove the pointer, leave the data

						if (sbHigh == key) {
							//We rarely end up in here, no real optimisation to be gained.
							#ifdef PRINT
							keyHit++;
							#endif
							js = oneMS;

							for (int jval = 0; jval < m; jval++) {
								if (js == sB) {
									output = (t + 1) * m - jval;
									//printf("t = %d, m = %d, jval = %d, output = %llu\n", t, m, jval, output);
									pointer = 0x0000FFFF;
									break;
								}

								js = montmul(js, newKB, mlo, mhi, mprimelo, mprimehi);

							}
							//printf("Match in S %d. t=%d, hash=%d, probe=%d beta=%llu rem=%d. Output will be %llu | %d*%d^%d-1\n", S, t, hash, probe, beta, remainder, b, thisk, *Base, ((output*Q) + remainder));

						}

						if (((pointer >= 32767) && first) || (pointer == 0x0000FFFF)) {
							break;
						}

						pointer = (pointer & 0x00007FFF);
						first = false;

					}

					#ifdef PRINT
					int endWLoop = clock();
					wloop_time_spent += (endWLoop - beginWLoop);
					#endif


				}

				#ifdef PRINT
				int endLoop = clock();
				loop_time_spent += (endLoop - beginLoop);
				#endif

				if (output > 0) {
					//printf("Output will be %llu | %d*%d^%d-1. Thread %d\n", b, thisk, base[0], ((output* *Q) + remainder), S);
					//printf("%llu | %d*%d^%d-1\n", b, thisk, base[0], ((output* *Q) + remainder));
     NOut[S] = thisk;
     NOut[SFull + S] = (output* *Q) + remainder;
					output = -1;
				}
			}
			#ifdef PRINT
			int endHash = clock();
			hash_time_spent += (endHash - beginHash);
			#endif	
		}
	}

	#ifdef PRINT
	int now = clock();
	time_spent = (now-end);
	if (printer) {
		printf("Cycles completing the big for loop was %d\n", time_spent);
	}
	#endif	

	#ifdef PRINT
	if (printer) {
		printf("Number of giant steps: %d\n", giant);
		//		printf("Number of collisions: %d\n", collisions);
		printf("Number of lookups against hash table was %d\n", lookups);
		printf("Max probe length was %d\n", maxProbe);
		printf("Number of hash key hits %d\n", keyHit);
		//		printf("Average probe length was %f\n", avgProbe / giant);
		//printf("Number of hits was %d\n", hits);
	}

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles to complete init k section was %d\n", init_time_spent);
		printf("Cycles to complete hash lookups was %d\n", hash_time_spent);
		printf("Cycles to complete for loop was %d\n", loop_time_spent);
		printf("Cycles to complete while loop was %d\n", wloop_time_spent);
		printf("Cycles to complete BSGS step was %d\n", time_spent);
		//		printf("Average (BSGS Cycles/muls) was %d\n", (time_spent / countmuls));
		printf("Average (BSGS Cycles/lookups) was %d\n", (time_spent / lookups));
	}
	#endif

	#ifdef PRINT
	time_spent = (end - beginfull);
	#endif

	#ifdef PRINT
	if (printer) {
		//printf("-------------------- Creating Hash Table --------------------\n");
		//printf("Cycles doing Bit Array Lookups (creating Hash Table) was %d (%d lookups @ %d cycles average) - %d%\n", bittime, bitLookups, bittime / bitLookups, (bittime * 100 / time_spent));
		//printf("Cycles doing Bit Array Updates (creating Hash Table) was %d (%d updates @ %d cycles average) - %d%\n", bitUtime, bitUpdates, bitUtime / bitUpdates, (bitUtime * 100 / time_spent));
		//printf("Cycles doing Hash Table Inserts was %d (%d inserts @ %d cycles average) - %d%\n", inserttime, inserts, inserttime / inserts, (inserttime * 100 / time_spent));
		//printf("Cycles doing Montgomery Multiplication was %d (%d function calls @ %d cycles average) - %d%\n", montmultime1, montmuls1, montmultime1 / montmuls1, (montmultime1 * 100 / time_spent));
		//printf("Total of components to create hash table is %d cycles\n", bittime+bitUtime+inserttime+montmultime1);
		//printf("--------------- Finished Creating Hash Table ----------------\n");
		printf("Cycles doing Legendre was %d (%d function calls @ %d cycles average) - %d%\n", legtime, leg, legtime / leg, (legtime * 100 / time_spent));
		//printf("Cycles doing Modul64 was %d (%d function calls @ %d cycles average) - %d%\n", modultime, modul, modultime / modul, (modultime * 100 / time_spent));
		//printf("Cycles doing Mulul64 was %d (%d function calls @ %d cycles average) - %d%\n", mulultime, mulul, mulultime / mulul, (mulultime * 100 / time_spent));
		//printf("Cycles doing Montgomery Multiplication was %d (%d function calls @ %d cycles average) - %d%\n", montmultime, montmuls, montmultime / montmuls, (montmultime * 100 / time_spent));
		printf("Cycles to execute one full thread was %d\n", time_spent);
	}
	#endif


}



int main(int argc, char* argv[])
{
	//Deal with command line arguments using getopt
	int inp;
	int threadScale = 1;
	int blockScale = 1;
	char *abcdFile = "sr_745.abcd";
 char *FactorFile = "factors.txt";

	unsigned long long low = 0;
	unsigned long long high = 0;
	int Qin = 0;
	int hashTableSize = 1;
	int verify = 1;
	int smallPrimeScale = 256;


	while ((inp = getopt(argc, argv, "b:hi:m:p:P:Q:O:s:t:v:")) != -1) {
		switch (inp) {
		case 'b':
			//Get the blockScale argument
			blockScale = strtol(optarg, NULL, 0);
			break;

		case 'h':
			//Print the help 
			cout << endl;
			cout << "CUDA Riesel Sieve 0.2.0 -- A sieve for multiple sequences of the form k*b^n-1" << endl;
			cout << endl;
			cout << "-i FILE  : Read in abcd sieve file called FILE." << endl;
			cout << "-m m     : Use m elements for the hash table." << endl;
			cout << "-p P0    : Start sieveing from P0. Must have a corresponding P1." << endl;
			cout << "-P P1    : Finish sieving at P1. Must have a corresponding P0. If no argument will use a scaled default for testing." << endl;
			cout << "                If no -p and -P arguments then we will default to 102297149770000 for testing." << endl;
			cout << "-Q Q     : Override subsequence value Q. Sieve k*b^n-1 as (k*b^d)*(b^Q)^m-1." << endl;
   cout << "-O FILE  : Write factors to a file called FILE" << endl;
			cout << "-b SCALE : Scale the number of CUDA blocks per kernel by the integer argument SCALE." << endl;
			cout << "-t SCALE : Scale the number of CUDA threads per block by the integer argument SCALE." << endl;
			cout << "                Note that these no longer require being powers of 2. We use the formula 1<<SCALE to ensure power of 2." << endl;
   cout << "-s SCALE : Scale the small primes by the integer argument SCALE. Default is 256." << endl;
			cout << "-v 	     : Don't verify factors. This implies running datless." << endl;
			cout << "                Switching off verification will lead to factors being found for candidates no longer in the sieve file due to running datless." << endl;
			cout << "                This will speed up sieving when large numbers of factors are being found, such as when using a small P0 value." << endl;    
			cout << "-h       : Prints this help" << endl;
			return 0;

		case 'i':
			//Get the input file argument
			abcdFile = optarg;
			//Check this is a valid filename at some point
			break;

		case 'm':
			hashTableSize = strtoull(optarg, NULL, 0);
			//Check this is large enough at some point. Can we still have problems if it is too small?
			break;

		case 'p':
			low = strtoull(optarg, NULL, 0);
			//Check this is large enough at some point. Can we still have problems if it is too small?
			break;

		case 'P':
			high = strtoull(optarg, NULL, 0);
			break;

		case 'Q':
			Qin = strtol(optarg, NULL, 0);
			if (Qin < 1 || Qin % 2 == 1) {
				cout << "Bad input parameter : Q must be an even integer greater than 0." << endl;
				return 1;
			}
			break;
   
  case 'O':
  	//Get the output file argument
			FactorFile = optarg;
			//Check this is a valid filename at some point
			break;

		case 's':
			smallPrimeScale = strtol(optarg, NULL, 0);
			break;

		case 't':
			//Get the threadScale argument
			threadScale = strtol(optarg, NULL, 0);
			break;

		case 'v': 
			//Turn off factor verification
			verify = 0;
			break;

		default:
			return 0;

		}
	}

	//Have a look at these limits in the future. We probably want to limit them more than this!
	if (blockScale < 1 || blockScale > 31) {
		cout << "Bad input parameter : 1 < BlockScale < 32. BlockScale is 1 by default, and should be a positive integer." << endl;
		return 1;
	}

	if (threadScale < 1 || threadScale > 31) {
		cout << "Bad input parameter : 1 < ThreadScale < 32. ThreadScale is 1 by default, and should be a positive integer." << endl;
		return 1;
	}

	if (hashTableSize < 1) {
		cout << "Bad input parameter. HashTableSize is 1 by default, and should be a positive integer." << endl;
		return 1;
	}

	cout << "BlockScale = " << blockScale << endl;
 cout << "ThreadScale = " << threadScale << endl;
 cout << "Low = " << low << endl;
 cout << "High = " << high << endl;
 cout << "Q_In = " << Qin << endl;
 cout << "HashTableSize = " << hashTableSize << endl;
 cout << "Verify = " << verify << endl;
 cout << "S = " << smallPrimeScale << endl;
 cout << "ABCD = " << abcdFile << endl;
 cout << "FactorFile = " << FactorFile << endl;
 cout << "" << endl;

 
	blockScale = 1 << blockScale - 1;
	threadScale = 1 << threadScale - 1;


	if (low == 0 && high == 0) {
		//Parameters were not set on the command line. Use our usual values for testing.

		//unsigned long long low = 600000;
		//unsigned long long high = 2500000000;

		//unsigned long long low = 6000000000;
		//unsigned long long high = 6004000000;

		//unsigned long long low = 1000067500000;
		//unsigned long long high = 1000070000000;

		//unsigned long long low = 1000099000000;
		//unsigned long long high = 1000100000000;

		//unsigned long long low = 102254819500000L;
		low = 102297149770000L;
		unsigned long long diff = 2500000L;
		high = low + (diff*threadScale*blockScale);
		//unsigned long long high = 102297160000000L;
	}

	if (high < low) {
		cout << "Bad input parameters : P1 < P0. We want to sieve primes p in the range P0 <= p <= P1." << endl;
		return 1;
	}


	const int blocks = 128 * blockScale;
	const int threads = 128 * threadScale;
	const int arraySize = blocks * threads;
	//This is based on the density of primes ~1/45 at 2^64. As we only consider odd numbers, 24 allows for roughly ~1/48 density, so we have some leeway here. We now scale this with the input value s. The less small primes we are using the smaller the testArray can be, and hence we gain some speed.
	int testArraySizeScaleFactor = 24;
	if (smallPrimeScale <= 4) {
		testArraySizeScaleFactor = 12; //Checked and seems to work for various b values
	}
 else 	if (smallPrimeScale <= 8) {
		testArraySizeScaleFactor = 13; //Test this more
	}
 else 	if (smallPrimeScale <= 16) {
		testArraySizeScaleFactor = 14; //Test this more
	}
 else 	if (smallPrimeScale <= 32) {
		testArraySizeScaleFactor = 16; //Not checked
	}
 else 	if (smallPrimeScale <= 64) {
		testArraySizeScaleFactor = 18; //Not checked
	}
 else 	if (smallPrimeScale <= 128) {
		testArraySizeScaleFactor = 20; //Not checked
	}
 else 	if (smallPrimeScale <= 256) {
		testArraySizeScaleFactor = 22; //Not checked
	}
 else {
  testArraySizeScaleFactor = 24; //Base case, should be fine up to 2^64 for any value of s
 } 
	const int testArraySize = arraySize * testArraySizeScaleFactor;
	const int hashScaling = 1;


	const int kb = 1024;
	const int mb = kb * kb;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);

	if (devCount < 1) {
		cout << "No CUDA enabled GPU was detected" << endl;
		return 1;
	}

	wcout << "CUDA Devices: " << devCount << endl;

	cudaDeviceProp props;

	for (int i = 0; i < devCount; ++i)
	{
		cudaGetDeviceProperties(&props, i);
		wcout << props.name << ":" << endl;
		wcout << "  CC: " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

		wcout << "  Warp size:         " << props.warpSize << endl;
		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		wcout << "  L2 Cache Size: " << props.l2CacheSize / kb << "kb" << endl;
		wcout << endl;
	}

	if (hashTableSize <= 1) {
		unsigned long long targetHashSize = 1;
		while (targetHashSize < props.totalGlobalMem / 2) {
			targetHashSize = targetHashSize << 1;
		}
		wcout << "Should target a " << targetHashSize / mb << "mb hash table" << endl << endl;


		//Use targetHashSize to set up the hash table - int = 32 bits = 4 bytes, so divide by 4
		//Each thread requires the a hash table, so also divide by arraySize
		long long longhashTableSize = (((targetHashSize / 4) / arraySize) / hashScaling);
		hashTableSize = longhashTableSize;
	}
	else {
		hashTableSize = 2 << hashTableSize;
	}

	cout << "Each thread should have " << hashTableSize * hashScaling << " buckets, to store " << hashTableSize << " elements. (Density 1/" << hashScaling << ")" << endl;



	//Read in an ABCD file and parse ----------------------------------------------------------------------------
	string line;
	int total = 0;
	//string abcdFile = "C:\\Users\\Rob\\Documents\\Visual Studio 2015\\Projects\\CPU Sieve\\sr_108.abcd";
	//string abcdFile = "C:\\Users\\Rob\\Desktop\\TestSieve\\sr_745.abcd";
	//string abcdFile = "sr_745.abcd";

	//First pass through the ABCD file to find the number of k's and max number of n's
	int count1 = 0; //Number of k's
	int count3 = 0; //Total number of lines

	countKs(abcdFile, count1, count3);

	//Second pass through the ABCD file to write the values into the matrix
	//Store the k and n values in this array
	//boost::numeric::ublas::matrix<int> kns(count1, max);
	//std::list<int> kns;
	count3 = count3 + (2 * count1);
	int *kns = (int *)malloc(count3 * sizeof(int));
	int *ks = (int *)malloc(count1 * sizeof(int));

	int minN = INT_MAX;
	int maxN = 0;

	//Reset the counts
	count1 = 0;
	count3 = 0;
	int baseCPU = 0;
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
				token = line.substr(n + 1);
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
				if (baseCPU == 0) {
					token = token.substr(n + 1);
					n = token.find("^");
					string b = token.substr(0, n);
					//cout << b << endl;
					baseCPU = stoi(b);
					cout << "The base is " << baseCPU << endl;
				}
				//Get the starting n-value - remove the square brakets
				n = token.find("[");
				token = token.substr(n + 1);
				n = token.find("]");
				token = token.substr(0, n);
				//cout << token << endl;
				total = stoi(token);
				if (total < minN) {
					minN = total;
				}
				kns[count3] = total;
				count3++;
				cout << "This is a new k-value with value " << kval << " and initial n-value " << total << endl;
			}
			else {
				//This is a number, n-value offset
				//cout << token << endl;
				int offset = stoi(token);
				total = total + offset;
				if (total > maxN) {
					maxN = total;
				}
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
	//-----------------------------------------------------------------------------------------------------------

	//Find the optimal Q value, and use subsequences with base b^Q ----------------------------------------------
	cout << "MinN = " << minN << " and MaxN = " << maxN << endl;
	cout << "Find the optimal(ish) Q value for subsequences in base b^Q" << endl;
	int range = maxN - minN;
	int minWork = count1 * range;
	int minSubs = count1;
	int minRange = range;
	int minQ = 1;
	cout << count1 << " k-values over a range of " << range << " n-values" << endl;

	int Qlow = 2;
	int Qhigh = 24;

	if (Qin != 0) {
		//Override these values for the look below
		Qlow = Qin;
		Qhigh = Qin + 1;
	}

	//Iterate through Q values - from 2 to 24(?)
	//At the moment, Q must be a multiple of 2 to work with the quadratic residues
	for (int Q = Qlow; Q < Qhigh; Q = Q + 2) {
		int subsequences = 0;
		//Count the number of subsequences
		bool *subseq = (bool *)malloc(count1*Q * sizeof(bool));
		memset(subseq, false, count1*Q * sizeof(bool));
		int whichk = -1;
		for (int qq = 0; qq < count3; qq++) {
			if (kns[qq] == 0) {
				whichk++;
				//Skip the k-value itself which is next in the array
				qq++;
			}
			else {
				//This is an n-value
				int mod = kns[qq] % Q;
				subseq[whichk*Q + mod] = true;
			}
		}
		//Count the number of true's in subseq - this is our number of subsequences if we use Q
		for (int qqq = 0; qqq < count1*Q; qqq++) {
			if (subseq[qqq] == true) {
				subsequences++;
			}
		}

		range = (maxN / Q) - (minN / Q) + 1;
		if (subsequences*range < minWork) {
			minQ = Q;
			minWork = subsequences * range;
			minSubs = subsequences;
			minRange = range;
		}
		cout << "Q=" << Q << ". Work= " << minWork << ". Range = " << range << ". Subsequences= " << subsequences << endl;
		free(subseq);
	}

	//Successfully found Q --------------------------------------------------------------------------------------
	if (Qin == 0) {
		cout << "Min Work has Q=" << minQ << ". Work= " << minWork << ". Range = " << minRange << ". Subsequences= " << minSubs << endl;
	}

	//Now we have minQ, we can calculate tMin and tMax to send to the gpu
	int shift = -1;
	int temp = hashTableSize;
	while (temp) {
		temp = temp >> 1;
		shift++;
	}
	int tMinCPU = (minN / minQ) >> shift;
	int tMaxCPU = (((maxN / minQ) + 1) >> shift) + 1;

	cout << "CPU thinks tMin = " << tMinCPU << " and tMax = " << tMaxCPU << endl;

	//Create a bit array that will tell us which n-values are interesting for each subsequence
	//minQ is the Q we will use, minSubs is the number of subsequences (rows in this matrix). range/32 is the number of ints to represent the range.
	int qRange = (minRange / 32) + 1;
	int rowoffset = qRange + 2;
	int *matrix = (int *)calloc(minSubs * rowoffset, sizeof(int));
	//The first column will contain the k, the second column will contain the remainder
	//Work through kns - if we find a new remainder add it to the next row
	int minimum = minN / minQ;
	cout << "Minimum = " << minimum << endl;
	int ak = 0;
	for (int qq = 0; qq < count3; qq++) {
		if (kns[qq] == 0) {
			qq++;
			ak = kns[qq];
		}
		else {
			//This is an n-value
			int n = kns[qq];
			int mod = n % minQ;
			//Check to see if this remainder is already in the matrix - if not add a new row for it

			//n should be rewritten in the form n=Qm+r. So subtract r and divide by Q to get m
			int m = ((n - mod) / minQ) - minimum;
			//Divide by 32 to find the correct bit, and do 31 - m (mod 32) to find the position
			int bit = m / 32;
			int pos = 31 - (m & 31);

			bool added = false;

			for (int rows = 0; rows < minSubs; rows++) {
				if ((matrix[rows*rowoffset] == ak) && (matrix[(rows*rowoffset) + 1] == mod)) {
					//This row has already been used, so set the correct bit to 1
					matrix[(rows*rowoffset) + bit + 2] += 1 << pos;
					added = true;
				}
			}
			if (added == false) {
				//The row didn't exist. Create it and add this element.
				for (int rows = 0; rows < minSubs; rows++) {
					if (matrix[rows*rowoffset] == 0) {
						//This row is empty
						matrix[rows*rowoffset] = ak;
						matrix[(rows*rowoffset) + 1] = mod;
						matrix[(rows*rowoffset) + bit + 2] += 1 << pos;
						break;
					}
				}
			}
		}
	}

	//for (int ddd = 0; ddd < (minSubs * rowoffset); ddd++) {
	//	if ((((ddd % rowoffset) == 0) || ((ddd % rowoffset) == 1)) && ddd<1200) {
	//		cout << (matrix[ddd]) << endl;
	//	}
	//}

	//cout << "-------------------------------------------------------------------------------------" << endl;

	//Can we sort the matrix before we send it to the GPU?
	int *matrix1 = (int *)calloc(minSubs * rowoffset, sizeof(int));
	for (int aaa = 0; aaa < minSubs; aaa++) {
		int kSearchmin = INT_MAX;
		int rSearchmin = INT_MAX;
		int kindex = 0;
		//Look through the matrix and find the smallest k and rem.
		for (int bbb = 0; bbb < minSubs; bbb++) {
			if ((matrix[bbb*rowoffset] <= kSearchmin) && (matrix[bbb*rowoffset] != INT_MAX)) {
				if (matrix[(bbb*rowoffset) + 1] < rSearchmin) {
					//This is the smallest k and r that we've found so far. Store the location
					kSearchmin = matrix[bbb*rowoffset];
					rSearchmin = matrix[(bbb*rowoffset) + 1];
					kindex = bbb;
				}
			}
		}
		//Copy the row now, and edit the k and r so we don't use them again
		for (int ccc = 0; ccc < rowoffset; ccc++) {
			matrix1[(aaa * rowoffset) + ccc] = matrix[(kindex * rowoffset) + ccc];
		}
		matrix[(kindex * rowoffset)] = INT_MAX;

		//for (int ddd = 0; ddd < (minSubs * rowoffset); ddd++) {
		//	if ((((ddd % rowoffset) == 0) || ((ddd % rowoffset) == 1)) && ddd<1200 && aaa<5) {
		//		cout << (matrix[ddd]) << endl;
		//	}
		//}
	}

	for (int ddd = 0; ddd < (minSubs * rowoffset); ddd++) {
		matrix[ddd] = matrix1[ddd];
		//if ((((ddd % rowoffset) == 0) || ((ddd % rowoffset) == 1)) && ddd<1200) {
		//	cout << (matrix[ddd]) << endl;
		//}
	}
	//matrix = matrix1;

	//Generate Primes -------------------------------------------------------------------------------------------

	//unsigned long long *KernelP = (unsigned long long *)malloc(arraySize * sizeof(unsigned long long));
	//int *NOut;
	unsigned int *KernelP;
	int *NOut = (int *)malloc(2 * arraySize * sizeof(int));
	cudaMallocHost((void **)&NOut, 2 * arraySize * sizeof(int));
	cudaMallocHost((void **)&KernelP, arraySize * sizeof(unsigned int));

	//Low should be greater than the primes we use below.


	unsigned long long startLow = low; //Don't touch this. Used for timing purposes

	//Use the idea of a segmented sieve. Generate a list of small primes first
	//If we use too many small primes then it can affect sieving, i.e. we cant use small -p values. Check for this in future
	clock_t begin = clock();
	int primeCount = 4096 * smallPrimeScale;
	int count = 0;
	bool *primes = (bool *)malloc(primeCount * 24 * sizeof(bool));
	unsigned int *smallP = (unsigned int *)malloc(primeCount * sizeof(unsigned int));
	memset(primes, true, primeCount * 24 * sizeof(bool));

	//First candidate will be 3, followed by each odd number in turn
	for (unsigned int i = 3; i < INT32_MAX; i += 2) {
		if (primes[i] == true) {
			smallP[count] = i;

			//Update count, and check to see if we are full
			count++;
			if (count == primeCount) {
				break;
			}

			//Now mark off all multiples of this in the boolean array
			for (int j = i * 2; j < primeCount * 24; j += i) {
				primes[j] = false;
			}
		}
	}
	//We could use this boolean array primes to generate a large mask. This would be useful when generating primes to send to the GPU.
	//If we default to primecount = 16384 * smallPrimeScale, we can use this to produce a mask for 3*5*7*11*13 = 15015 rows and capture the cyclicity.
	//Or we could just manually produce a mask using the small primes which will probably be easier.

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Time generating small primes " << time_spent << "s" << endl;


	unsigned long long *mark1 = (unsigned long long *)malloc(((testArraySize / 64) + 1) * sizeof(unsigned long long));
	//testArraySize is 393216*blockScale*threadScale. This is divided by 64 as numbers are represented as bits, so 6144 longs. 
	//3*5*7*11 = 1155, so use that as the mask, as it will always fit, though we could in theory use a larger mask, it doesn't need to cycle around.
	
	//We are now looking at using 4 masks of roughly equal size. This will help performance while also keeping memory usage lower than using masks consisting of sequential primes
	//We need to update these with the correct primes, up to and including 61.
	//The groups are: 3,5,19,37,47    11,17,43,61    13,23,31,53    7,29,41,59
	
	unsigned long long *mask = (unsigned long long *)malloc(sizeof(unsigned long long) * 3 * 5 * 19 * 37 * 47);
	memset(mask, 0, 3*5*19*37*47 * sizeof(unsigned long long));
	
	unsigned long long *mask1 = (unsigned long long *)malloc(sizeof(unsigned long long) * 11 * 17 * 43 * 61);
	memset(mask1, 0, 11*17*43*61 * sizeof(unsigned long long));
	
	unsigned long long *mask2 = (unsigned long long *)malloc(sizeof(unsigned long long) * 13 * 23 * 31 * 53);
	memset(mask2, 0, 13*23*31*53 * sizeof(unsigned long long));
	
	unsigned long long *mask3 = (unsigned long long *)malloc(sizeof(unsigned long long) * 7 * 29 * 41 * 59);
	memset(mask3, 0, 7*29*41*59 * sizeof(unsigned long long));
	
	//Assume the first element =0 mod 1155
	int limit = 3 * 5 * 19 * 37 * 47 * 64;
	for (int k = 0; k < limit; k += 3) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask[intbool] = mask[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 5) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask[intbool] = mask[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 19) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask[intbool] = mask[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 37) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask[intbool] = mask[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 47) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask[intbool] = mask[intbool] | (0x8000000000000000 >> intshift);
	}


	//Mask1
	limit = 11 * 17 * 43 * 61 * 64;
	for (int k = 0; k < limit; k += 11) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask1[intbool] = mask1[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 17) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask1[intbool] = mask1[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 43) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask1[intbool] = mask1[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 61) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask1[intbool] = mask1[intbool] | (0x8000000000000000 >> intshift);
	}


	//Mask2
	limit = 13 * 23 * 31 * 53 * 64;
	for (int k = 0; k < limit; k += 13) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask2[intbool] = mask2[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 23) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask2[intbool] = mask2[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 31) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask2[intbool] = mask2[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 53) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask2[intbool] = mask2[intbool] | (0x8000000000000000 >> intshift);
	}
	
	
	//Mask3
	limit = 7 * 29 * 41 * 59 * 64;
	for (int k = 0; k < limit; k += 7) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask3[intbool] = mask3[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 29) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask3[intbool] = mask3[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 41) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask3[intbool] = mask3[intbool] | (0x8000000000000000 >> intshift);
	}
	for (int k = 0; k < limit; k += 59) {
		//Take k and divide it by 64 to work out which int we are in. Shift by the remainder
		int intbool = k / 64; //Find the right int
		int intshift = k & 63; //Work out the shift

		mask3[intbool] = mask3[intbool] | (0x8000000000000000 >> intshift);
	}


	//for (int k = 0; k < 1155; k++) {
	//	cout << mask[k] << endl;
	//}


	//Try setting up the GPU just once

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}
	cudaStatus = cudaSetDevice(0);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	// Allocate GPU buffers for three vectors (one output, seven input). 
	//Give all vectors same size for now, we can change this afterwards
 
 cudaStatus = cudaMalloc((void**)&dev_a, 2 * arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_c, minSubs * rowoffset * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_f, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_g, arraySize * hashTableSize * hashScaling * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_h, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_i, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_j, ((arraySize * hashTableSize * hashScaling) / 32) * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_k, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_n, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//Copy the data to the correct GPU buffers

	cudaStatus = cudaMemcpy(dev_c, matrix, minSubs * rowoffset * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << (count1 * 3 + minSubs + 3) * sizeof(int) << "bytes" << endl;
		//goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol(base, &baseCPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed for base!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(dev_f, &rowoffset, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(dev_h, &hashTableSize, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(dev_i, &hashScaling, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(dev_k, &minQ, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol(tMin, &tMinCPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed for tMin!");
		//goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol(tMax, &tMaxCPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed for tMax!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(dev_n, &minSubs, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	//Create the first set of primes for the GPU before we start
	if (low % 2 == 0) {
		//Make sure low is odd. Go back by one if necessary
		low = low - 1;
		cout << "We've reduced low by 1 to make it odd" << endl;
	}

	cout << "Low is now set to " << low << endl;
	generateGPUPrimes(KernelP, low, smallP, testArraySize, primeCount, arraySize, mark1, mask, mask1, mask2, mask3);
	cout << endl;

	int kernelCount = 0;
 int factorCount = 0;
	//clock_t loopTime = clock();
	auto loopTime = std::chrono::high_resolution_clock::now();
	//From here we need to loop to keep the GPU busy.

	int *KernelPOld = (int *)malloc(arraySize * sizeof(int)); 

	cudaStream_t stream0;
	cudaStreamCreate(&stream0);


	while (low < high) {

		auto begin = std::chrono::high_resolution_clock::now();
		kernelCount++;
		//cout << "Executing kernel number " << kernelCount << endl;

		cudaStatus = cudaMemcpyToSymbolAsync(lowGPU, &low, sizeof(unsigned long long),0,cudaMemcpyHostToDevice,stream0);
		//cudaStatus = cudaMemcpyToSymbol(lowGPU, &low, sizeof(unsigned long long));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy to constant memory failed for lowGPU!");
			//goto Error;
		}

		//KernelP = KernelP1;

		unsigned int minPrime = KernelP[0];
		unsigned int maxPrime = KernelP[arraySize - 1];
		unsigned int progress = maxPrime - minPrime;

		//cout << "Min Prime = " << minPrime  + low << ". Max Prime = " << maxPrime + low << ". Progress = " << progress << endl;
		//cout << "Array Size = " << arraySize << endl;

		//Set low to the next odd number above maxPrime. The CPU will generate the next batch of primes while the GPU is working
  unsigned long long oldLow = low;
		low = low + maxPrime + 2;


		//cout << "Try to launch the CUDA kernel" << endl;

		cudaEvent_t start, stop;
		cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
		cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
		// Launch a kernel on the GPU with one thread for each element.
		cudaEventRecord(start, stream0);


		cudaMemcpyAsync(dev_b, KernelP, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice, stream0);
  cudaMemsetAsync(dev_a, 0, 2 * arraySize * sizeof(int), stream0);
		cudaMemsetAsync(dev_g, 0, arraySize * hashTableSize * hashScaling * sizeof(int), stream0);
		cudaMemsetAsync(dev_j, 0, ((arraySize * hashTableSize * hashScaling) / 32) * sizeof(unsigned int), stream0);
		addKernel1 <<<blocks, threads, 0, stream0 >>>(dev_a, dev_b, dev_c, dev_f, dev_g, dev_h, dev_i, dev_j, dev_k, dev_n);
		//cudaStreamSynchronize(stream0);
		//We should try to generate the next array of primes in here!
		

		cudaEventRecord(stop, stream0);
  //unsigned int *KernelPOld;
		if (low < high) {
  memcpy (KernelPOld, KernelP, arraySize * sizeof(unsigned int));
   
		generateGPUPrimes(KernelP, low, smallP, testArraySize, primeCount, arraySize, mark1, mask, mask1, mask2, mask3);
		}
		//cudaEventQuery(stop);
		//cudaDeviceSynchronize();
		cudaEventSynchronize(stop);
		cudaStreamSynchronize(stream0);

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(NOut, dev_a, 2 * arraySize * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy output failed!");
  }
  
  for (int i=0; i<arraySize; i++) {
    if (NOut[arraySize+i] > 0) {    
      if (verify == 1) {
        //We should verify this factor before adding it to the output. 
        //This involve two steps, make sure the 'n' value is in the sieve file and then check the factor is correct.
        //As the sieve runs datless most factors will not be useful as the 'n' value won't be in the sieve file.
        unsigned int nVal = NOut[arraySize+i];
        unsigned int kVal = NOut[i];
        unsigned long long factorCandidate = KernelPOld[i] + oldLow;
        
        //Check the sieve file
        for (int k=0; k<count3; k++) {
          //Search for a 0
          if (kns[k] == 0) {
            if (kns[k+1] == kVal) {
              //We've found the correct k, now loop through the n values
              int x = 1;
              int y = k+2;
              while (x>0) {
                x = kns[y];
                y++;
                if (x == nVal) {
                 //We've had a match, so this n-value was in the original sieve file
                 //Check the factor is valid
                 
                 unsigned __int128 val = baseCPU;
                 for (int j=1; j<nVal; j++){
                  val = val*baseCPU;
                  val = val%factorCandidate;
                 }

                 val = val*kVal;
                 val = val%factorCandidate;
                 //Subtracting 1 should leave 0, so check that val is 1.
                 if (val == 1) {
                   //Factor is valid
                   cout << factorCandidate  << " | " << kVal << "*" << baseCPU << "^" << nVal << "-1" << endl;
                   writeFactorToFile(factorCandidate, kVal, baseCPU, nVal, FactorFile);
                   factorCount++;
                 }
                 break;
                }
                if (x > nVal) {
                 //We've already gone past where this value would have been, so can give up.
                 break;
                }
              }
            }
          }
        }
      }
      
      else if (verify == 0) {
        //Just print out the factor to the screen and save it in the factor file.
        cout << KernelPOld[i] + oldLow  << " | " << NOut[i] << "*" << baseCPU << "^" << NOut[arraySize+i] << "-1" << endl;
        writeFactorToFile(KernelPOld[i] + oldLow, NOut[i], baseCPU, NOut[arraySize+i], FactorFile);
        factorCount++;
      }
      
      else {
       //We should never get here, verify should only ever be 0 or 1.
       cout << "Error: Verify has an unexpected value: " << verify << endl; 
      }
    }
  }

  
		//end = clock();
		//time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		//cout << "Time to execute kernel (outside function) " << time_spent << "s" << endl;
		//cout << "Progress = " << progress << " at " << progress / time_spent << " p/sec" << endl;
		//cout << "" << endl;
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_spent = (end - begin);
		//cout << "Time to execute kernel (outside function) " << time_spent.count() << "s" << endl;
		//cout << "Progress = " << progress << " at " << progress / time_spent.count() << " p/sec" << endl;
		double percent = (((double)((low - 2) - startLow) / (high-startLow)))*100;
		std::chrono::duration<double> timeSoFar = (end - loopTime);
		double timeLeft = ((double)100.0/percent * timeSoFar.count()) - timeSoFar.count();
		//std::chrono::time_point<std::chrono::system_clock> ETA += std::chrono::seconds(timeLeft);
		std::time_t ETA = std::chrono::system_clock::to_time_t(end + std::chrono::seconds((int)timeLeft)); 

		if (percent > 100) {
			percent = 100;
			timeLeft = 0;
		}
		//printf("%.2f%% done, Remaining time %.2fs. ",percent, timeLeft);

  //We should update this to run every X seconds where X is user configurable. It is only a minimum time between outputs as the kernel may take longer than X.
		if (kernelCount % 20 == 0) {
		  char *ETAnice = strtok(ctime(&ETA), "\n");
		  printf("p=%llu, %.0f p/sec,  %.2f%% done, %d factors found, ETA %s \n", low, progress / time_spent.count(), percent, factorCount, ETAnice);
		  //fflush(stdout);
		}



		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Something went wrong in running the kernel");
			//goto Error;
		}

	}

	//clock_t loopEnd = clock();
	//time_spent = (double)(loopEnd - loopTime) / CLOCKS_PER_SEC;
	//cout << "Time taken " << time_spent << "s" << endl;
	//cout << "Time per kernel " << time_spent / kernelCount << endl;
	//cout << "Progress = " << (low - 2) - startLow << " at " << ((low - 2) - startLow) / time_spent << " p/sec" << endl << endl;
	auto loopEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> loop_time_spent = (loopEnd - loopTime);
	cout << endl;
	cout << "Time taken: " << loop_time_spent.count() << "s" << endl;
 cout << "Factors found: " << factorCount << " at " << loop_time_spent.count() / factorCount << " seconds per factor"  << endl;
	cout << "Time per kernel: " << loop_time_spent.count() / kernelCount << endl;
	cout << "Progress = " << (low - 2) - startLow << " at " << ((low - 2) - startLow) / loop_time_spent.count() << " p/sec" << endl << endl;


	//Reprint the CUDA info
	//wcout << "CUDA version:   v" << CUDART_VERSION << endl;

	cudaGetDeviceCount(&devCount);
	//wcout << "CUDA Devices: " << devCount << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaGetDeviceProperties(&props, i);
		wcout << props.name << ":" << endl;
		wcout << "  CC: " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl;
		wcout << "  L2 Cache Size: " << props.l2CacheSize / kb << "kb" << endl;
		wcout << endl;
	}

	//cout << "Each thread used " << hashTableSize * hashScaling << " buckets, to store " << hashTableSize << " elements. (Density 1/" << hashScaling << ")" << endl;
	//cout << "Hash table size was " << (hashTableSize*hashScaling * 4 * arraySize) / mb << "mb of GPU RAM" << endl;
	//cout << "Blocksize = " << blocks << ". Threads per block = " << threads << "." << endl;

Error:
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_f);
	cudaFree(dev_g);
	cudaFree(dev_h);
	cudaFree(dev_i);
	cudaFree(dev_j);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


void writeFactorToFile(unsigned long long p, unsigned int k, unsigned int b, unsigned int n, char *factorFile) {
 //Factors will be of the form p | k*b^n-1
 FILE *file = fopen(factorFile, "a");
 fprintf(file, "%llu | %d*%d^%d-1\n", p, k, b, n);
 fclose(file);
}

void generateGPUPrimes(unsigned int *KernelP1, unsigned long long low, unsigned int *smallP, int testArraySize, int primeCount, int arraySize, unsigned long long *mark1, unsigned long long *mask, unsigned long long *mask1, unsigned long long *mask2, unsigned long long *mask3) {
//	clock_t begin = clock();
	unsigned int diff = 0;

	//for (int i = 0; i < ((testArraySize / 32) + 1); i++) {
	//	mark1[i] = 0;
	//}

	//Lets deal with 3 and 5 using a mask
	//unsigned long long mask[15];
	//mask[0] = 0x96692cd259a4b349;
	//mask[8] = 0x6692cd259a4b3496;
	//mask[1] = 0x692cd259a4b34966;
	//mask[9] = 0x92cd259a4b349669;
	//mask[2] = 0x2cd259a4b3496692;
	//mask[10] = 0xcd259a4b3496692c;
	//mask[3] = 0xd259a4b3496692cd;
	//mask[11] = 0x259a4b3496692cd2;
	//mask[4] = 0x59a4b3496692cd25;
	//mask[12] = 0x9a4b3496692cd259;
	//mask[5] = 0xa4b3496692cd259a;
	//mask[13] = 0x4b3496692cd259a4;
	//mask[6] = 0xb3496692cd259a4b;
	//mask[14] = 0x3496692cd259a4b3;
	//mask[7] = 0x496692cd259a4b34;

	//unsigned int offset = low % 15015;
	//offset = (offset * 12317) % 15015;


	//for (int i = 0; i < ((testArraySize / 64) + 1); i++) {
	//	mark1[i] = mask[offset];
		//cout << mark1[i] << endl;
	//	offset++;;
	//	if (offset > 15014) {
	//		offset = offset - 15015;
	//	}
	//}


	//Mask 3*5*7*11*13*17*19
//	unsigned long long offset = low % 4849845;
//	offset = (offset * 3751052) % 4849845;


//	for (int i = 0; i < ((testArraySize / 64) + 1); i++) {
//		mark1[i] = mask[offset];
		//cout << mark1[i] << endl;
//	 offset++;;
//		if (offset > 4849844) {
//			offset = offset - 4849845;
//		}
//	}

	//Mask 23*29*31*37
//	offset = low % 765049;
//	offset = (offset * 328732) % 765049;


//	for (int i = 0; i < ((testArraySize / 64) + 1); i++) {
//		mark1[i] = mark1[i] | mask1[offset];
		//cout << mark1[i] << endl;
//		offset++;;
//		if (offset > 765048) {
//			offset = offset - 765049;
//		}
//	}
 
 //Can we combine these two masks together quickly??
 	//The groups are: 3,5,19,37,47    11,17,43,61    13,23,31,53    7,29,41,59

	//Mask 3*5*7*11*13*17*19
	//unsigned long long offset = low % 4849845;
	//offset = (offset * 3751052) % 4849845;
 
 //Mask 23*29*31*37
	//unsigned long long offset1 = low % 765049;
	//offset1 = (offset1 * 328732) % 765049;
 
 	//for (int i = 0; i < ((testArraySize / 64) + 1); i++) {
		//mark1[i] = mask[offset] | mask1[offset1];
  //offset++;
		//if (offset > 4849844) {
		//	offset = offset - 4849845;
		//}
		//offset1++;
		//if (offset1 > 765048) {
		//	offset1 = offset1 - 765049;
		//}
	//}
 
 //Mask 3*5*19*37*47
 unsigned long long offset = low % 495615;
	offset = (offset * 3872) % 495615;
 
 //Mask 11*17*43*61
 unsigned long long offset1 = low % 490501;
	offset1 = (offset1 * 195434) % 490501;
 
 //Mask 13*23*31*53
 unsigned long long offset2 = low % 491257;
	offset2 = (offset2 * 211087) % 491257;
 
 //Mask 7*29*41*59
 unsigned long long offset3 = low % 491057;
	offset3 = (offset3 * 180310) % 491057;
 
 
 for (int i = 0; i < ((testArraySize / 64) + 1); i++) {
	  mark1[i] = mask[offset] | mask1[offset1] | mask2[offset2] | mask3[offset3];
   offset++;
		 if (offset > 495614) {
			  offset = offset - 495614;
		 }
		 offset1++;
		 if (offset1 > 490500) {
		   offset1 = offset1 - 490500;
		 }
   offset2++;
		 if (offset2 > 491256) {
			  offset2 = offset2 - 491256;
		 }
		 offset3++;
		 if (offset3 > 491056) {
		   offset3 = offset3 - 491056;
		 }
	}
 
 
 
 

	//Lets deal with multiples of 3 separately using a mask.
	//unsigned int mask[3];
	//mask[0] = 0x92492492;
	//mask[1] = 0x49249249;
	//mask[2] = 0x24924924;

	//unsigned int offset = low % 3;


	//for (int i = 0; i < ((testArraySize / 32) + 1); i++) {
	//	mark1[i] = mask[(offset + i) % 3];
	//}

	//unsigned int mask1[5];
	//mask1[0] = 0x84210842;
	//mask1[1] = 0x10842108;
	//mask1[2] = 0x42108421;
	//mask1[3] = 0x08421084;
	//mask1[4] = 0x21084210;

	//offset = (5 - (low % 5))%5;

	//for (int i = 0; i < ((testArraySize / 32) + 1); i++) {
	//	mark1[i] = mark1[i] | mask1[(offset + i) % 5];
	//}

//	clock_t end = clock();
//	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	cout << "Time setting the masks " << time_spent << "s" << endl;

 
 
 //This section deals will all small primes that are not in the masks above
 
//	begin = clock();
// unsigned int totalTouches = 0;
 
 int intbool = 0;
 int intshift = 0;
// unsigned int touch = 0;
 //int changes = 0;
 
	//Start at 11 now as we deal with 3,5,7,11,13,17,19,23,29,31,37 with masks.
	//#pragma omp parallel for num_threads(4)
	for (int i = 17; i < primeCount; i++) {
		unsigned int smallPrime = smallP[i];
		unsigned int mod = low % smallPrime;

		if (mod == 0) {
			diff = 0;
		}
		else {
			if (mod % 2 == 1) {
				mod = mod + smallPrime;
			}
			mod = mod >> 1;
			diff = (smallPrime - mod);
			//diff = (smallPrime * (2 - (mod & 1)) - mod ) >> 1;
		}

		for (int k = diff; k < testArraySize; k += smallPrime) {
			//Take k and divide it by 32 to work out which int we are in. Shift by the remainder
   intbool = k >> 6; //Find the right int. This is slightly faster than division by 64.
			intshift = k & 63; //Work out the shift
			
   //int num_one_bits = __builtin_popcountl(mark1[intbool]);
			mark1[intbool] = mark1[intbool] | (0x8000000000000000 >> intshift);
//			touch++;
   //int new_num_one_bits = __builtin_popcountl(mark1[intbool]);
   
   //if (num_one_bits < new_num_one_bits) {
   // changes++;
   //}

			//if (((mark1[intbool] >> (31 - intshift)) & 1) == 1) {
				//Do nothing - this bit is already 1
			//}
			//else {
			//	mark1[intbool] += (0x80000000 >> intshift);
			//}
		}
//		if (smallPrime < 64) {
//			cout << smallPrime << ": " << touch << endl;
//		}
//		totalTouches += touch;
//		touch = 0;

	}

//	cout << totalTouches << endl;
 //cout << "Changes made marking prime array: " << changes << endl;

//	end = clock();
//	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	cout << "Time marking the prime array " << time_spent << "s" << endl;

//	begin = clock();
	//Use mark1 to decide if we need to add a prime. We're looking for 0 entries in these ints
	int countPrimes = 0;
// int lasti = 0;
 
	//for (int i = 0; i < testArraySize; i++) {
	//	int intbool = i / 64; //Find the right int
	//	int intshift = i & 63; //Work out the shift
	//	if (((mark1[intbool] >> (63 - intshift)) & 1) == 0) {
 
 //cout << testArraySize << testArraySize/64 << endl;
 for (int i = 0; i< testArraySize/64; i++) {
   //int num_zero_bits = 64 - __builtin_popcountl(mark1[i]);
   //cout << "Number of prime candidates in this long: " << num_zero_bits << endl;
   //cout << "Check entry " << i << endl;
   for (int j = 0; j<64; j++) { 
     if (((mark1[i] >> (63 - j)) & 1) == 0) {
			    KernelP1[countPrimes] = 2 * (i*64 + j);
			    countPrimes++;
       		  
       if (countPrimes == arraySize) {
				   //cout << "We got as far as " << 2*i + low << " out of " << low + (testArraySize*2) << endl;
//       lasti = i;
				   break;
       }
       
       //Prime candidates are not very dense, so we look for an early break in this loop. 
       //Once a prime candidate has been added to the list we set the mark to 1, and check to see if all marks are now 1 by comparing with max value for ulonglong.
       mark1[i] = mark1[i] | (0x8000000000000000 >> j);
       if (mark1[i] == 18446744073709551615) {
         break;
       }
			  }
	  }
  			if (countPrimes == arraySize) {
				//cout << "We got as far as " << 2*i + low << " out of " << low + (testArraySize*2) << endl;
				break;
    }
	}
 
 if (countPrimes < arraySize) {
   cout << "Error: Not enough primes in the buffer to send to GPU. Must increase the constants that control the value of testArraySize" << endl;
   cout << "countPrimes = " << countPrimes << ". arraySize = " << arraySize << endl;
   exit(1);
 }
 
// cout << "Used " << lasti << " out of " << testArraySize/64 << " elements of the testArray " << (100*lasti)/(testArraySize/64) << "%" << endl;  

//	end = clock();
//	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//	cout << "Time generating kernel primes " << time_spent << "s" << endl;

}

