//Complete - Read in an ABCD file, find Q, split into subsequences
//Complete - CUDA code with correct outputs

//TODO - Scale hash table to the size of GPU RAM


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>

#include "FileInput.h"

using namespace std;

void generateGPUPrimes(unsigned long long *KernelP, unsigned long long low, unsigned int *smallP, int testArraySize, int primeCount, int arraySize, unsigned int *mark1);

#define PRINT true
//#undef PRINT

int *dev_a = 0; //NOut
unsigned long long *dev_b = 0; //KernelP
int *dev_c = 0; //kns
				//__constant__ int dev_c[512];
__constant__ int base[1]; //Base
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
		//t = x, x = y, y = t;
		x = x ^ y;
		y = x ^ y;
		x = x ^ y;

		if (x % 4 == 3 && y % 4 == 3) {
			sign++;
		}
	}

	return sign & 1;
}


__device__  __forceinline__ void xbinGCDnew(unsigned long long beta, unsigned long long &u, unsigned long long &v)
{
	unsigned long long alpha = 9223372036854775808;
	//unsigned long long a = 9223372036854775808;
	//u = 1; v = 0;
	//alpha = a;
	// Note that alpha is
	// even and beta is odd.
	// The invariant maintained from here on is: 2a = u*2*alpha - v*beta.

	//while (a > 0) { //This is just a counter as a is never used. 
	//	a = a >> 1;
	#pragma unroll 1
	for (int i=0; i<64; i++) {
		if ((u & 1) == 0) { // Delete a common
			u = u >> 1; v = v >> 1; // factor of 2 in
		} // u and v.
		else {
			/* We want to set u = (u + beta) >> 1, but
			that can overflow, so we use Dietz's method. */
			u = ((u ^ beta) >> 1) + (u & beta);
			v = (v >> 1) + alpha; //v>>1 happens in both cases, this just also sets the highest bit to 1
		}
	}
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

	unsigned int tm0 = 0;
	unsigned int tm1 = 0;

	//We need to calculate all 128 bits of t = abar * bbar. Add straight to u
	//PTX Version 1
	asm("mul.hi.u32 %1, %6, %8;\n\t" //Bits 32-64 (u1)
		"mul.lo.u32 %2, %7, %9;\n\t" //Bits 65-96 (u2)
		"mul.lo.u32 %0, %6, %8; \n\t" //Lowest 32 bits of u (u0)
		"mad.lo.cc.u32 %1, %6, %9, %1;\n\t" //Add the crossproduct to u1. Set the carry out
		"madc.hi.cc.u32 %2, %6, %9, %2;\n\t" //Add the crossproduct to u2. Set the carry out, and use the carry in
		"madc.hi.u32 %3, %7, %9, %3;\n\t" //Bits 97-128 plus any carry out (u3)
		"mad.lo.cc.u32 %1, %7, %8, %1;\n\t" //Add the other crossproduct to u1. Set the carry out
		"madc.hi.cc.u32 %2, %7, %8, %2;\n\t" //Add the other crossproduct to u2. Set the carry out, and use the carry in
		"addc.u32 %3, %3, 0;\n\t" //Add the potential carry into u3

	//Now we calculate t*mprime & mask. I.e. the low 64 bits
		"mul.hi.u32 %5, %0, %12;\n\t" //tm1
		"mul.lo.u32 %4, %0, %12;\n\t" //tm0 = lo part of u0*mprimelo
		"mad.lo.u32 %5, %0, %13, %5;\n\t" //Add the lo part of the crossproduct to tm1. u0*mprimehi
		"mad.lo.u32 %5, %1, %12, %5;\n\t" //Add the lo part of the other crossproduct to tm1 u1*mprimelo

	//Multiple tm*m to get the 128 bit product, and add it to u
		"mad.lo.cc.u32 %0, %4, %10, %0;\n\t" //Add the lo part of tm.lo*m*lo to u0. Set the carry out
		"madc.hi.cc.u32 %1, %4, %10, %1;\n\t" //Add the hi part of tm.lo*m*lo to u1. Set the carry out, and use the carry in
		"madc.lo.cc.u32 %2, %5, %11, %2;\n\t" //Add the lo part of tm.hi*m*hi to u2. Set the carry out, and use the carry in
		"madc.hi.cc.u32 %3, %5, %11, %3;\n\t" //Add the hi part of tm.hi*m*hi to u3. Use the carry in, and set carry out for overflow detection
		"addc.u32 %0, 0, 0;\n\t" //Put the overflow flag into u0
		"mad.lo.cc.u32 %1, %4, %11, %1;\n\t" //Add the lo part of tm.lo*m*hi to u1. Set the carry out
		"madc.hi.cc.u32 %2, %4, %11, %2;\n\t" //Add the hi part of tm.lo*m*hi to u2. Set the carry out, and use the carry in
		"addc.cc.u32 %3, %3, 0;\n\t" //Add the potential carry into u3
		"addc.u32 %0, %0, 0;\n\t" //Put the overflow flag into u0
		"mad.lo.cc.u32 %1, %5, %10, %1;\n\t" //Add the lo part of tm.hi*m*lo to u1. Set the carry out
		"madc.hi.cc.u32 %2, %5, %10, %2;\n\t" //Add the hi part of tm.hi*m*lo to u2. Set the carry out, and use the carry in
		"addc.cc.u32 %3, %3, 0\n\t;" //Add the potential carry into u3
		"addc.u32 %0, %0, 0;" //Put the overflow flag into u0

		: "=+r"(u0), "=+r"(u1), "=+r"(u2), "=+r"(u3), "=+r"(tm0), "=+r"(tm1) : "r"(alo), "r"(ahi), "r"(blo), "r"(bhi), "r"(mlo), "r"(mhi), "r"(mprimelo), "r"(mprimehi)
	);

	//We're only interested in u2 and u3
	unsigned long long u = u3;
	u = u << 32;
	u = u | u2;

	unsigned long long m = mhi;
	m = m << 32;
	m = m | mlo;

	if (u0 > 0 || u >= m) {
		u = u - m;
	}
	return u;

}

__device__ __forceinline__ long long binExtEuclid(long long a, long long b) {
	long long u = b;
	long long v = a;
	long long r = 0;
	long long s = 1;
	long long x = a;
	while (v>0) {
		if ((u & 1) == 0) {
			u = u >> 1;
			r = (r + ((r & 1)*b)) >> 1;
			//if ((r & 1) == 0) {
			//	r = r >> 1;
			//}
			//else {
			//	r = (r + b) >> 1;
			//}
		}
		else {
			if ((v & 1) == 0) {
				v = v >> 1;
				s = (s + ((s & 1)*b)) >> 1;
				//if ((s & 1) == 0) {
				//	s = s >> 1;
				//}
				//else {
				//	s = (s + b) >> 1;
				//}
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



__global__ void addKernel1(int *NOut, unsigned long long *KernelP, int *knmatrix, int *rowOffset, int *hashKeys, int *hashElements, int *hashDensity, unsigned int *bits, int *Q, int *minSubs)
{
	clock_t beginfull = clock();
	clock_t begin = clock();

	int legtime = 0;
	clock_t beginLeg = clock();
	clock_t endLeg = clock();

	clock_t end = clock();
	int time_spent = 0;

	//This deals with the hashTables
	const int m = *hashElements;
	const int mem = m * (*hashDensity); //This is hashTableElements*density, to keep the correct thread using correct hash table 

	//m=2^shift, calculate shift
	int shift = 31 - __clz(m);

	const int S = (blockIdx.x * blockDim.x) + threadIdx.x; //This is this block ID*threads in a block + threadID
	const int Sm = S * mem;
	const int Sints = Sm >> 5;

	bool printer = false;
	if (S == 0) {
		printer = true;
	}

	const unsigned long long b = KernelP[S];
	const unsigned long long oneMS = modul64(1, 0, b);

	unsigned long long bprime = 0;
	unsigned long long rInv = 1;

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

	xbinGCDnew(b, rInv, bprime);

	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		//Check GCD has worked
		printf("2*inp*rInv - b*bprime = %llu\n", (2 * 9223372036854775808 * rInv - (b*bprime)));
		printf("Cycles to do xbinGCD was %d\n", time_spent);
	}
	#endif

	//beginModul = clock();
	unsigned long long KernelBase = modul64(base[0], 0, b);
	unsigned long long newKB = oneMS;

	//endModul = clock();
	//time_spent = (endModul - beginModul);
	//modultime += time_spent;
	//modul = modul + 2;

	unsigned int mlo = (unsigned int)b;
	unsigned int mhi = (unsigned int)(b >> 32);
	unsigned int mprimelo = (unsigned int)bprime;
	unsigned int mprimehi = (unsigned int)(bprime >> 32);

	//We now deal with b^Q for subsequences. 
	for (int qq = 0; qq < *Q; qq++) {
		//beginMont = clock();
		newKB = montmul(KernelBase, newKB, mlo, mhi, mprimelo, mprimehi);
		//newKB = montmul64(KernelBase, newKB, b, bprime);
		//endMont = clock();
		//time_spent = (endMont - beginMont);
		//montmultime += time_spent;
		//montmuls++;
	}

	//Save this now so we can use it later. Will save a call to modul
	unsigned long long c1 = newKB;

	unsigned long long plo = newKB * rInv;
	unsigned long long phi = __umul64hi(newKB, rInv);


	//beginModul = clock();
	newKB = modul64(phi, plo, b);
	//endModul = clock();
	//time_spent = (endModul - beginModul);
	//modultime += time_spent;
	//modul++;

	//unsigned long long newKB2 = newKB;

	#ifdef PRINT
	begin = clock();
	#endif

	newKB = binExtEuclid(newKB, b);

	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles to do binExtEuclid was %d\n", time_spent);
	}
	#endif


	unsigned long long js = oneMS;

	//beginModul = clock();
	//Convert js to montgomery space
	//js = modul64(js, 0, b);
	//Convert newKB back into Montgomery space
	newKB = modul64(newKB, 0, b);
	//endModul = clock();
	//time_spent = (endModul - beginModul);
	//modultime += time_spent;
	//modul = modul + 2;


	//Do a dry run through the baby steps to find the free positions in the hash table
	#ifdef PRINT
	begin = clock();
	#endif

	unsigned int hash = 0;

	for (int j = 0; j < m; j++) {

		hash = (int)js & (m - 1);

		//if ((bits[Sints + (hash / 32)] & (1 << (hash & 31))) == 0) {
		//hashKeys[(Sm + hash1)] = (js & 0xFFFF0000) + 0x0000FFFF;
		//	bits[Sints + (hash / 32)] += (1 << (hash & 31));
		//}
		bits[Sints + (hash >> 5)] |= (1 << (hash & 31));

		js = montmul(js, newKB, mlo, mhi, mprimelo, mprimehi);
		//js = montmul64(js, newKB, b, bprime);

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
	//int hash = 0;
	int store = 0;
	int key = 0;
	int pointer = 0;
	int firstFree = 0; //The first memory cell that doesn't head a linked list
	//int bitArrayCounter = 0;

	for (int j = 0; j < m; j++) {

		hash = (int)js & (m - 1);
		store = (js & 0xFFFF0000) + 0x0000FFFF; //This blanks off the last 16 bits and adds a null pointer. This will contain our pointer

		key = hashKeys[(Sm + hash)];

		//if (((bits[Sints + (hash / 32)] >> (hash&31)) & 1) == 1) {
		//if (key == 0) {
		//You were the element stored here, so subtract the 1
		//store = store;
		//bits[Sints + (hash / 32)] -= (1 << (hash & 31));
		//}

		//else {
		if (key != 0) {
			//You were a collision into this bucket. Find somewhere to live, and update the pointer
			//key = hashKeys[(Sm + hash)];

			//while ((hashKeys[Sm + firstFree] != 0) || (((bits[Sints + (firstFree / 32)] >> (firstFree & 31)) & 1) == 1)) {
			while (((bits[Sints + (firstFree >> 5)] >> (firstFree & 31)) & 1) == 1) {
				firstFree++;
			}
			//Could we do this with bfind instead?

			//for (int i = bitArrayCounter; i < (m >> 5); i++) {
			//	if (bits[Sints + i] != 0xFFFFFFFF) {
					//Find the position of the first zero, and turn it into a 1
			//		int pos = __clz(bits[Sints + i]);

			//		break;
			//	}
			//	bitArrayCounter = i;
			//}

			hashKeys[(Sm + firstFree)] = ((store - 0x0000FFFF) + (key & 0x0000FFFF)); //Store this new data, with the pointer from the head. We're now 2nd in this linked list
			store = (key & 0xFFFF0000) + firstFree + m;
			firstFree++;

		}

		hashKeys[(Sm + hash)] = store; //Update the linked list head, either with new data and a null pointer, or an updated pointer

		js = montmul(js, newKB, mlo, mhi, mprimelo, mprimehi);
		//js = montmul64(js, newKB, b, bprime);


	}


	//Finished calculating the hash table --------------------------------------------------------------------

	////Try to populate our new hash table array ------------------------------------------------------------
	////New method - first m buckets (0 to (m-1)) are the beginning of linked lists. Buckets m to (2m-1) are for the collisions
	//int lookups = 0;
	//int hash = 0;
	//int firstFree = m; //The first memory cell that doesn't head a linked list
	//int store = 0;
	//int key = 0;
	//int pointer = 0;

	//for (int j = 0; j<m; j++) {

	//	hash = js & (m - 1);
	//	store = js & 0xFFFF0000; //This blanks off the last 16 bits. This will contain our pointer

	//							 //beginLoop = clock();

	//							 //beginInsert = clock();

	//							 //int key = hashKeys[(Sm + hash)];

	//	if ((bits[Sints + (hash / 32)] & (1 << (hash & 31))) == 0) {
	//		//if (set == 0) {
	//		//if (key == 0) {
	//		//This linked list contains nothing yet, so add the element, and a zero pointer
	//		key = store;
	//		bits[Sints + (hash / 32)] += 1 << (hash & 31);
	//	}

	//	//if (set != 0) {
	//	else {
	//		key = hashKeys[(Sm + hash)];

	//		//This linked list has at least one element in it. Copy the pointer, put that in our data.
	//		pointer = key & 0x0000FFFF; //This removes the top 16 bits which contain the data, just leaves the pointer.
	//		key = (key & 0xFFFF0000) + firstFree; //Update the original data with the new pointer to this data

	//											  //We could gather these up in shared memory and write them out every so often. 
	//		hashKeys[(Sm + firstFree)] = (store + pointer); //Store this new data, with the pointer from the head. We're now 2nd in this linked list

	//		firstFree++; //Update the location of next free memory cell
	//	}


	//	//beginInsert = clock();
	//	hashKeys[(Sm + hash)] = key;
	//	//endInsert = clock();
	//	//time_spent = (endInsert - beginInsert);
	//	//inserttime += time_spent;
	//	//inserts++;

	//	//beginMont = clock();
	//	js = montmul(js, newKB, b, bprime);
	//	//endMont = clock();
	//	//time_spent = (endMont - beginMont);
	//	//montmultime += time_spent;
	//	//montmultime1 += time_spent;

	//	//montmuls++;
	//	//montmuls1++;

	//}


	////Finished calculating the hash table --------------------------------------------------------------------

	#ifdef PRINT
	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles calculating new hash table was %d (%d inserts (baby steps) @ %d cycles average)\n", time_spent, m, time_spent / m);
	}
	#endif

	begin = clock();
	//Compute KernelBase^-m (mod b)
	//unsigned long long c1 = modul64(newKB2, 0, b);
	//modul++;

	//c1  should be KernelBase^Q^-1 (mod b) computed earlier
	//Now repeatedly square it as m is a power of two

	for (int t = 0; t < shift; t++) {
		//beginMont = clock();
		c1 = montmul(c1, c1, mlo, mhi, mprimelo, mprimehi);
		//c1 = montmul64(c1, c1, b, bprime);
		//endMont = clock();
		//time_spent = (endMont - beginMont);
		//montmultime += time_spent;
		//montmuls++;
	}

	int output = -1;

	//int tMin = tMin[0];
	//int tMax = tMax[0];

	#ifdef PRINT
	if (printer) {
		printf("tMin = %d. tMax = %d\n", tMin[0], tMax[0]);
	}
	#endif

	lookups = 0;
	int countmuls = tMin[0];
	int giant = 0;
	//	int collisions = 0;

	int maxProbe = 0;
	//	float avgProbe = 0;

	unsigned long long fixedBeta = 0;
	unsigned long long beta = oneMS;

	for (int t = 0; t < tMin[0]; t++) {
		//beginMont = clock();
		beta = montmul(beta, c1, mlo, mhi, mprimelo, mprimehi);
		//beta = montmul64(beta, c1, b, bprime);
		//endMont = clock();
		//time_spent = (endMont - beginMont);
		//montmultime += time_spent;
		//montmuls++;
	}

	int leg1;
	int leg2fixed = legendre(base[0], b);
	int leg2;
	int leg = 1;

	int probe = 0;

	//int hits = 0;

	int thisk = 0;
	int corek = 0;
	int lastk = 0;
	int remainder = 0;

	//Work through the matrix of kn values
	for (int k = 0; k < *minSubs; k++) {
		//while (k < *minSubs) {
			lastk = thisk;
			thisk = knmatrix[k* *rowOffset];
			if (thisk != lastk) {
				beginLeg = clock();
				leg1 = legendre(thisk, b);
				leg2 = !(leg1^leg2fixed); //Rather than use k*base, use the multiplicative property of legendre to save any overflows
#ifdef PRINT
				leg = leg + 1;
				endLeg = clock();
				time_spent = (endLeg - beginLeg);
				legtime += time_spent;
#endif		
				fixedBeta = modul64(thisk, 0, b);
				fixedBeta = montmul(fixedBeta, beta, mlo, mhi, mprimelo, mprimehi);
				//fixedBeta = montmul64(fixedBeta, beta, b, bprime);
			}

			remainder = knmatrix[(k* *rowOffset) + 1];

			//if ((remainder % 2 == 0 && leg1 == 1) || (remainder % 2 == 1 && leg2 == 1)) {
			//	break;
			//}
			//k++;
		//}

		if ((remainder % 2 == 0 && leg1 == 1) || (remainder % 2 == 1 && leg2 == 1)) {
			//We need to do something
			unsigned long long sB = fixedBeta;
			for (int rem = 0; rem < remainder; rem++) {
				sB = montmul(sB, KernelBase, mlo, mhi, mprimelo, mprimehi);
				//sB = montmul64(sB, KernelBase, b, bprime);
			}

			for (int t = tMin[0]; t < tMax[0]; t++) {
				giant++;

				//Check if beta is in js
				//hash = sB & (m - 1);

				probe = 0;
				pointer = (int)sB & (m - 1);

				//while (true) {

				//	//This was quicker with the bit array in the past - it now appears to be faster without using the bit array
				//	int key = hashKeys[(Sm + pointer)];
				//	lookups++;

				//	probe++;
				//	if (probe > maxProbe) {
				//		maxProbe = probe;
				//	}

				//	pointer = key & 0x0000FFFF; //Remove the data, leave the pointer
				//	key = key & 0xFFFF0000; //Remove the pointer, leave the data

				//	if (((int)sB & 0xFFFF0000) == key) {

				//		unsigned long long jsnew = oneMS;

				//		for (int jval = 0; jval < m; jval++) {
				//			if (jsnew == sB) {
				//				output = t * m + jval;
				//				pointer = 0;
				//				break;
				//			}

				//			jsnew = montmul(jsnew, newKB, b, bprime);

				//		}
				//		//printf("Match in S %d. t=%d, hash=%d, probe=%d beta=%llu rem=%d. Output will be %llu | %d*%d^%d-1\n", S, t, hash, probe, beta, remainder, b, thisk, *Base, ((output*Q) + remainder));

				//	}

				//	if (pointer == 0) {
				//		break;
				//	}

				//}



				while (true) {

					//Surely this must be faster on some cards - it will save us some memory transactions!
					//if ((((bits[Sints + (hash / 32)] >> (hash & 31)) & 1) == 0) && probe == 0) {
					//	break;
					//}

					//This was quicker with the bit array in the past - it now appears to be faster without using the bit array
					key = hashKeys[(Sm + pointer)];
					lookups++;

					//if (printer && k == 6) {
					//	printf("Probe = %d. Data = %d. Pointer = %d.\n", probe, (key & 0xFFFF0000), (key & 0x0000FFFF));
					//}

					probe++;
					if (probe > maxProbe) {
						maxProbe = probe;
					}

					pointer = (key & 0x0000FFFF); //Remove the data, leave the pointer
					key = key & 0xFFFF0000; //Remove the pointer, leave the data

					if (((int)sB & 0xFFFF0000) == key) {

						js = oneMS;

						for (int jval = 0; jval < m; jval++) {
							if (js == sB) {
								output = t * m + jval;
								pointer = 0x0000FFFF;
								break;
							}

							js = montmul(js, newKB, mlo, mhi, mprimelo, mprimehi);
							//js = montmul64(js, newKB, b, bprime);

						}
						//printf("Match in S %d. t=%d, hash=%d, probe=%d beta=%llu rem=%d. Output will be %llu | %d*%d^%d-1\n", S, t, hash, probe, beta, remainder, b, thisk, *Base, ((output*Q) + remainder));

					}

					if (pointer == 0x0000FFFF) {
						break;
					}

					pointer = pointer - m;

					//if (probe > 15) {
					//printf("The pointer was %d\n", pointer);
					//	if (probe > 17) {
					//		printf("Killed by probe length\n");
					//			for (int i = 0; i < m; i++) {
					//				printf("Hash[%d] = Data: %d, Pointer: %d\n", i, (hashKeys[Sm + i]) & 0xFFFF0000, (hashKeys[Sm + i]) & 0x0000FFFF);
					//			}
					//		break;
					//	}
					//}

				}

				sB = montmul(sB, c1, mlo, mhi, mprimelo, mprimehi);
				//sB = montmul64(sB, c1, b, bprime);


			}
			if (output > 0) {
				printf("Output will be %llu | %d*%d^%d-1. Thread %d\n", b, thisk, base[0], ((output* *Q) + remainder), S);
				output = -1;
			}
		}

		//if (printer) {
		//printf("Lookups by k %d = %d\n", k, lookups);
		//}
	}

	#ifdef PRINT
	if (printer) {
		printf("Number of giant steps: %d\n", giant);
		//		printf("Number of collisions: %d\n", collisions);
		printf("Number of lookups against hash table was %d\n", lookups);
		printf("Max probe length was %d\n", maxProbe);
		//		printf("Average probe length was %f\n", avgProbe / giant);
		//printf("Number of hits was %d\n", hits);
	}

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles to complete BSGS step was %d\n", time_spent);
		//		printf("Average (BSGS Cycles/muls) was %d\n", (time_spent / countmuls));
		printf("Average (BSGS Cycles/lookups) was %d\n", (time_spent / lookups));
	}
	#endif

	//begin = clock();

	//NOut[S] = output; //This should contain the k-value in the top 32 bits and the n-value in the low 32 bits

	//end = clock();
	//time_spent = (end - begin);
	//if (printer) {
	//	printf("Cycles to write output to NOut was %d\n", time_spent);
	//}


	time_spent = (end - beginfull);
	//NOut[S] = time_spent;
	NOut[S] = lookups;
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

	unsigned long long low = 0;
	unsigned long long high = 0;
	int Qin = 0;
	int hashTableSize = 1;


	while ((inp = getopt(argc, argv, "b:hi:m:p:P:Q:t:")) != -1) {
		switch (inp) {
		case 'b':
			//Get the blockScale argument
			blockScale = strtol(optarg, NULL, 0);
			break;

		case 'h':
			//Print the help 
			cout << endl;
			cout << "CUDA Riesel Sieve 0.1.0 -- A sieve for multiple sequences of the form k*b^n-1" << endl;
			cout << endl;
			cout << "-i FILE  : Read in abcd sieve file called FILE." << endl;
			cout << "-m m     : Use m elements for the hash table." << endl;
			cout << "-p P0    : Start sieveing from P0. Must have a corresponding P1." << endl;
			cout << "-P P1    : Finish sieving at P1. Must have a corresponding P0. If no argument will use a scaled default for testing." << endl;
			cout << "             If no -p and -P arguments then we will default to 102297149770000 for testing." << endl;
			cout << "-Q Q     : Override subsequence value Q. Sieve k*b^n-1 as (k*b^d)*(b^Q)^m-1." << endl;
			cout << "-b SCALE : Scale the number of CUDA blocks per kernel by the integer argument SCALE." << endl;
			cout << "-t SCALE : Scale the number of CUDA threads per block by the integer argument SCALE." << endl;
			cout << "             Note that these no longer require being powers of 2. We use the formula 1<<SCALE to ensure power of 2." << endl;
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

		case 't':
			//Get the threadScale argument
			threadScale = strtol(optarg, NULL, 0);
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

	cout << "Trying to launch with BlockScale = " << blockScale << " and ThreadScale = " << threadScale << endl;

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
	const int testArraySize = arraySize * 24;
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
					//ifstream myfile(abcdFile);
					//if (myfile.is_open())
					//{
					//	while (getline(myfile, line))
					//	{
					//		count3++;

					//			string::size_type n = line.find(" ");
					//		string token = line.substr(0, n);
					//cout << token << endl;

					//If tokens[0] == "ABCD" then this defines a new k, otherwise it is a number
					//		if (token.compare("ABCD") == 0) {
					//			count1++;
					//cout << "We're here!" << endl;
					//		}
					//	}
					//	myfile.close();
					//}

					//else cout << "Unable to open file first time" << endl;

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
					cout << "The base is " << base << endl;
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

	//Print out the bit array
	//for (int rows = 0; rows <1; rows++) {
	//cout << matrix[rows*rowoffset] << endl;
	//cout << matrix[(rows*rowoffset) + 1] << endl;
	//cout << matrix[(rows*rowoffset) + 2] << endl;
	//cout << matrix[(rows*rowoffset) + 3] << endl;
	//cout << matrix[(rows*rowoffset) + 4] << endl;
	//}


	//Generate Primes -------------------------------------------------------------------------------------------

	//unsigned long long *KernelP = (unsigned long long *)malloc(arraySize * sizeof(unsigned long long));
	int *NOut;
	unsigned long long *KernelP;
	//int *NOut = (int *)malloc(arraySize * sizeof(int));
	cudaMallocHost((void **)&NOut, arraySize * sizeof(int));
	cudaMallocHost((void **)&KernelP, arraySize * sizeof(unsigned long long));

	//Low should be greater than the primes we use below.


	unsigned long long startLow = low; //Don't touch this. Used for timing purposes

	//Use the idea of a segmented sieve. Generate a list of small primes first
	//If we use too many small primes then it can affect sieving, i.e. we cant use small -p values. Check for this in future
	clock_t begin = clock();
	int primeCount = 16384 * 128;
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
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Time generating small primes " << time_spent << "s" << endl;


	unsigned int *mark1 = (unsigned int *)malloc(((testArraySize / 32) + 1) * sizeof(unsigned int));


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

	cudaStatus = cudaMalloc((void**)&dev_c, minSubs * rowoffset * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//cudaStatus = cudaMalloc((void**)&dev_e, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMalloc((void**)&dev_f, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_g, arraySize * hashTableSize * hashScaling * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_h, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_i, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_j, ((arraySize * hashTableSize) / 32) * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_k, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//cudaStatus = cudaMalloc((void**)&tMin, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&tMax, sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMalloc((void**)&dev_n, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Copy the data to the correct GPU buffers

	//Lets try storing the k values and remainders in constant memory instead
	//cudaStatus = cudaMemcpyToSymbol(dev_c, ks2, (count1 * 3 + minSubs + 3) * sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy to constant memory failed!");
	//	cout << (count1 * 3 + minSubs + 3) * sizeof(int) << "bytes" << endl;
	//	goto Error;
	//}
	cudaStatus = cudaMemcpy(dev_c, matrix, minSubs * rowoffset * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaMemcpy to constant memory failed!");
		cout << (count1 * 3 + minSubs + 3) * sizeof(int) << "bytes" << endl;
		goto Error;
	}

	//cudaStatus = cudaMemcpy(dev_e, &base, sizeof(int), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMemcpyToSymbol(base, &baseCPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_f, &rowoffset, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_h, &hashTableSize, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_i, &hashScaling, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_k, &minQ, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//cudaStatus = cudaMemcpy(tMin, &tMin, sizeof(int), cudaMemcpyHostToDevice); //minN
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMemcpyToSymbol(tMin, &tMinCPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol(tMax, &tMaxCPU, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed!");
		goto Error;
	}

	//cudaStatus = cudaMemcpy(tMax, &tMax, sizeof(int), cudaMemcpyHostToDevice); //maxN
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	cudaStatus = cudaMemcpy(dev_n, &minSubs, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//cudaStatus = cudaMemcpy(dev_j, bits, ((arraySize * hashTableSize * hashScaling) / 32) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	//Create the first set of primes for the GPU before we start
	if (low % 2 == 0) {
		//Make sure low is odd. Go back by one if necessary
		low = low - 1;
		cout << "We've reduced low by 1 to make it odd" << endl;
	}

	cout << "Low is now set to " << low << endl;
	generateGPUPrimes(KernelP, low, smallP, testArraySize, primeCount, arraySize, mark1);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int kernelCount = 0;
	clock_t loopTime = clock();
	//From here we need to loop to keep the GPU busy. 
	while (low < high) {

		kernelCount++;
		cout << "Executing kernel number " << kernelCount << endl;

		unsigned long long minPrime = KernelP[0];
		unsigned long long maxPrime = KernelP[arraySize - 1];
		unsigned long long progress = maxPrime - minPrime;

		cout << "Min Prime = " << minPrime << ". Max Prime = " << maxPrime << ". Progress = " << progress << endl;
		cout << "Array Size = " << arraySize << endl;

		//Set low to the next odd number above maxPrime. The CPU will generate the netx batch of primes while the GPU is working
		low = maxPrime + 2;


		begin = clock();
		cout << "Try to launch the CUDA kernel" << endl;

		cudaStream_t stream0;
		//cudaStream_t stream1;
		cudaStreamCreate(&stream0);
		//cudaStreamCreate(&stream1);

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpyAsync(dev_b, KernelP, arraySize * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy input failed!");
		}

		cudaStatus = cudaMemsetAsync(dev_g, 0, arraySize * hashTableSize * hashScaling * sizeof(int), stream0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy input failed!");
		}

		cudaStatus = cudaMemsetAsync(dev_j, 0, ((arraySize * hashTableSize) / 32) * sizeof(unsigned int), stream0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy input failed!");
		}

		cudaStreamSynchronize(stream0);

		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		// Launch a kernel on the GPU with one thread for each element.
		//cudaEventRecord(start);


		addKernel1 << <blocks, threads, 0, stream0 >> >(dev_a, dev_b, dev_c, dev_f, dev_g, dev_h, dev_i, dev_j, dev_k, dev_n);
		//addKernel1 << <blocks, threads, 0, stream1 >> >(dev_a, dev_b, dev_c, dev_e, dev_f, dev_g, dev_h, dev_i, dev_j);

		//This uses too much shared memory and kills occupancy. Really we want to use no more than 16 ints per thread (for 64 threads per block)!
		//addKernel1 << <blocks, threads, ((threads*hashElements*hashDensity) / 32)*sizeof(int), stream0 >> >(dev_a, dev_b, dev_c, dev_e, dev_f, dev_g, dev_h, dev_i);

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

		//We should try to generate the next array of primes in here!
		generateGPUPrimes(KernelP, low, smallP, testArraySize, primeCount, arraySize, mark1);

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpyAsync(NOut, dev_a, arraySize * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy output failed!");
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			cudaGetLastError();
		}


		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "Time to execute kernel (outside function) " << time_spent << "s" << endl;

		printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			NOut[0], NOut[32], NOut[64], NOut[96], NOut[128], NOut[160], NOut[192], NOut[224], NOut[256], NOut[288]);

		int maxCycles = 0;
		int maxThreadID = 0;
		int minCycles = INT_MAX;
		int minThreadID = 0;
		for (int i = 0; i < arraySize; i++) {
			if (NOut[i] > maxCycles) {
				maxCycles = NOut[i];
				maxThreadID = i;
			}
			if (NOut[i] < minCycles) {
				minCycles = NOut[i];
				minThreadID = i;
			}
		}

		cout << "Max Cycles in this kernel was " << maxCycles << " in Thread " << maxThreadID << endl;
		cout << "Min Cycles in this kernel was " << minCycles << " in Thread " << minThreadID << endl;

		long long totalCycles = 0;
		for (int i = 0; i < arraySize; i++) {
			totalCycles += NOut[i];
		}

		cout << "Total cycles for this kernel was " << totalCycles << " @ " << totalCycles / (arraySize) << " cycles average" << endl << endl;

	}

	clock_t loopEnd = clock();
	time_spent = (double)(loopEnd - loopTime) / CLOCKS_PER_SEC;
	cout << "Time taken " << time_spent << "s" << endl;
	cout << "Time per kernel " << time_spent / kernelCount << endl;
	cout << "Progress = " << (low - 2) - startLow << " at " << ((low - 2) - startLow) / time_spent << " p/sec" << endl << endl;

	//Reprint the CUDA info
	wcout << "CUDA version:   v" << CUDART_VERSION << endl;

	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << devCount << endl;

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

	cout << "Each thread used " << hashTableSize * hashScaling << " buckets, to store " << hashTableSize << " elements. (Density 1/" << hashScaling << ")" << endl;
	cout << "Hash table size was " << (hashTableSize*hashScaling * 4 * arraySize) / mb << "mb of GPU RAM" << endl;
	cout << "Blocksize = " << blocks << ". Threads per block = " << threads << "." << endl;

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	//cudaFree(dev_e);
	cudaFree(dev_f);
	cudaFree(dev_g);
	cudaFree(dev_h);
	cudaFree(dev_i);
	cudaFree(dev_j);
	//return cudaStatus;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

void generateGPUPrimes(unsigned long long *KernelP, unsigned long long low, unsigned int *smallP, int testArraySize, int primeCount, int arraySize, unsigned int *mark1) {
	clock_t begin = clock();
	//unsigned int *mark1 = (unsigned int *)malloc(((testArraySize / 32) + 1)*sizeof(unsigned int));
	//memset(mark1, 0, ((testArraySize / 32) + 1)*sizeof(unsigned int));
	unsigned int diff = 0;

	//Lets deal with multiples of 3 separately using a mask.
	unsigned int mask[3];
	mask[0] = 0x92492492;
	mask[1] = 0x49249249;
	mask[2] = 0x24924924;

	unsigned int offset = low % 3;

	for (int i = 0; i < ((testArraySize / 32) + 1); i++) {
		mark1[i] = mask[(offset + i) % 3];
	}

	//unsigned int mask1[5];
	//mask1[0] = 0x84210842;
	//mask1[1] = 0x42108421;
	//mask1[2] = 0x21084210;
	//mask1[3] = 0x10842108;
	//mask1[4] = 0x08421084;

	//offset = low % 5;

	//for (int i = 0; i < ((testArraySize / 32) + 1); i++) {
	//	mark1[i] = mark1[i] | mask1[(offset + i) % 5];
	//}

	//1000 0100 0010 0001 0000 1000 0100 0010

	//Start at 2 now as we deal with 3 and 5 with masks
	for (int i = 1; i < primeCount; i++) {
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
		}

		for (int k = diff; k < testArraySize; k += smallPrime) {
			//mark[k] = false;
			//Take k and divide it by 32 to work out which int we are in. Shift by the remainder
			int intbool = k / 32; //Find the right int
			int intshift = k & 31; //Work out the shift
			if (((mark1[intbool] >> (31 - intshift)) & 1) == 1) {
				//Do nothing - this bit is already 1
			}
			else {
				mark1[intbool] += (0x80000000 >> intshift);
			}
		}

	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Time marking the prime array " << time_spent << "s" << endl;

	begin = clock();
	//Use mark1 to decide if we need to add a prime. We're looking for 1 entries in these ints
	int countPrimes = 0;
	for (int i = 0; i < testArraySize; i++) {
		int intbool = i / 32; //Find the right int
		int intshift = i & 31; //Work out the shift
		if (((mark1[intbool] >> (31 - intshift)) & 1) == 0) {
			KernelP[countPrimes] = 2 * i + low;
			countPrimes++;
			if (countPrimes == arraySize) {
				cout << "We got as far as " << i + low << " out of " << low + (testArraySize) << endl;
				break;
			}
		}
	}

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Time generating kernel primes " << time_spent << "s" << endl;

}




