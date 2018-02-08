//Complete - Read in an ABCD file, find Q, split into subsequences
//Complete - CUDA code with correct outputs

//TODO - Scale hash table to the size of GPU RAM


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

cudaError_t addWithCuda(int *NOut, unsigned long long *KernelP, unsigned int size, const int blocks, const int threads, int *hashKeys, int hashElements, int hashDensity, unsigned int *bits);
int *dev_a = 0; //NOut
unsigned long long *dev_b = 0; //KernelP
//int *dev_c = 0; //kns
__constant__ int dev_c[512];
int *dev_e; //Base
int *dev_f; //counterIn
int *dev_g = 0; //HashTable Keys
int *dev_h = 0; //HashTableElements
int *dev_i = 0; //HashTableDensity
unsigned int *dev_j = 0;

cudaError_t cudaStatus;


__device__  __forceinline__ void xbinGCDnew(unsigned long long a, unsigned long long beta, unsigned long long &u, unsigned long long &v)
{
	unsigned long long alpha;
	u = 1; v = 0;
	alpha = a;
	// Note that alpha is
	// even and beta is odd.
	// The invariant maintained from here on is: 2a = u*2*alpha - v*beta.

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
}


__device__ __forceinline__ void mulul64new(unsigned long long u, unsigned long long v, unsigned long long &wlo, unsigned long long &whi)
{
	unsigned long long u0, u1, v0, v1, k, t;
	unsigned long long w0, w1, w2;

	u1 = u >> 32; u0 = u & 0xFFFFFFFF;
	v1 = v >> 32; v0 = v & 0xFFFFFFFF;

	t = u0*v0;
	w0 = t & 0xFFFFFFFF;
	k = t >> 32;

	t = u1*v0 + k;
	w1 = t & 0xFFFFFFFF;
	w2 = t >> 32;

	t = u0*v1 + w1;
	k = t >> 32;

	wlo = (t << 32) + w0;
	whi = u1*v1 + w2 + k;

}

__device__ __forceinline__ unsigned long long modul64(unsigned long long x, unsigned long long y, unsigned long long z) {
	/* Divides (x || y) by z, for 64-bit integers x, y,
	and z, giving the remainder (modulus) as the result.
	Must have x < z (to get a 64-bit result). This is
	checked for. */
	long long t;
	if (x >= z) {
		printf("Bad call to modul64, must have x < z.");
	}
	for (int i = 1; i <= 64; i++) { // Do 64 times.
		t = (long long)x >> 63; // All 1's if x(63) = 1.
		x = (x << 1) | (y >> 63); // Shift x || y left      <- Bitwise OR?
		y = y << 1; // one bit.
		if ((x | t) >= z) {
			x = x - z;
			y = y + 1;
		}
	}
	return x; // Quotient is y.
}

__device__ __forceinline__ unsigned long long montmul(unsigned long long abar, unsigned long long bbar, unsigned long long m, unsigned long long mprime) {
	unsigned long long thi, tlo, tm;
	//unsigned long long uhi, ulo;
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
	//tmmhi = __umul64hi(tm, m);
	//tmmlo = tm*m;


	//PTX Version 2 - Clobbers less registers - very similar speed.
	asm(//"{.reg .u64 t1;\n\t"              // temp 64-bit reg t1 = tmmlo
		"mad.lo.cc.u64 %0, %3, %4, %0;\n\t" //MAD: "tlo = (tm*m) + tlo" and set the carry out 
		//"add.cc.u64 %0, %0, t1;\n\t" //Add tlo = tlo + tmmlo and set carry out. 
		"madc.hi.cc.u64 %1, %3, %4, %1;\n\t" //MAD: "thi = hi(tm*m) + thi" use the previous carry and set the carry out
		//"addc.cc.u64 %1, %1, %3;\n\t" //Add thi and tmmhi, use the previous carry and set carry out.
		"addc.u32 %2, 0, 0;" //This sets ov to 1 if the previous addition overflowed.
		//"}"
		: "=+l"(tlo), "=+l"(thi) "=r"(ov) : "l"(tm), "l"(m)
		);

	if (ov > 0 || thi >= m) // If u >= m,
		thi = thi - m; // subtract m from u.
	return thi;
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



__global__ void addKernel1(int *NOut, unsigned long long *KernelP, /*int *ks,*/ int *Base, int *counterIn, int *hashKeys, int *hashElements, int *hashDensity, unsigned int *bits)
{
	clock_t beginfull = clock();
	clock_t begin = clock();

	//This deals with the hashTables
	const int m = *hashElements;
	const int mem = m * (*hashDensity); //This is hashTableElements * 4 to reduce collisions. Must be a power of 2
	const int ints = mem / 32;

	int shift = 0;
	int tempM = m;
	//m=2^shift, calculate shift
	while (tempM > 1) {
		tempM = tempM >> 1;
		shift++;
	}


	/*unsigned int bitArray[mem/32]; //Bit array for hash table

	for (int ii = 0; ii < 64; ii++) {
		bitArray[ii] = 0;
	}*/

	int S = (blockIdx.x * blockDim.x) + threadIdx.x; //This is this block ID*threads in a block + threadID
	int Sm = S*mem;

	bool printer = false;
	if (S == 0) {
		printer = true;
	}

	//extern __shared__ int bitArray[];

	//for (int ii = 0; ii < 64; ii++) {
	//	bitArray[threadIdx.x * 64 + ii] = 0;
	//}

	unsigned long long b = KernelP[S];
	int Q = dev_c[2];
	int NMin = dev_c[0]/Q;
	int NMax = (dev_c[1]/Q)+1;

	unsigned long long bprime = 0;
	unsigned long long rInv = 0;
	
	int montmuls = 0;
	int modul = 0;
	int mulul = 0;


	clock_t end = clock();
	int time_spent = (end - begin);
	if (printer) {
		printf("KernelBase = %d\n", *Base);
		printf("HashTableElements = %d. %d at 1/%d density.\n", mem,m,*hashDensity);
		printf("Each Thread should use %d ints in its bit array.\n", ints);
		printf("Q = %d, NMin = %d, NMax = %d\n", Q,NMin, NMax);
		printf("Cycles to complete variable setup was %d\n", time_spent);
	}

	begin = clock();

	xbinGCDnew(9223372036854775808, b, rInv, bprime);

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		//Check GCD has worked
		printf("2*inp*rInv - b*bprime = %llu\n", (2 * 9223372036854775808 * rInv - (b*bprime)));
		printf("Cycles to do xbinGCD was %d\n", time_spent);
	}


	unsigned long long KernelBase = modul64(*Base, 0, b);
	modul++;
	unsigned long long newKB = modul64(1, 0, b);
	modul++;
	//We now deal with b^Q for subsequences. 
	for (int qq = 0; qq < Q; qq++) {
		newKB = montmul(KernelBase, newKB, b, bprime);
		montmuls++;
	}

	unsigned long long plo = 0;
	unsigned long long phi = 0;

	begin = clock();

	mulul64new(newKB, rInv, plo, phi);
	mulul++;

	end = clock();
	time_spent = (end - begin);

	if (printer) {
		printf("Cycles to do mulul64 was %d\n", time_spent);
	}

	newKB = modul64(phi, plo, b);
	modul++;
	
	unsigned long long newKB2 = newKB;

	begin = clock();

	newKB = binExtEuclid(newKB, b);

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles to do binExtEuclid was %d\n", time_spent);
	}

	begin = clock();
	//Try to populate our new hash table array ------------------------------------------------------------
	int lookups = 0;
	int hash = 0;
	unsigned long long js = 1;

	//Convert js to montgomery space
	js = modul64(js, 0, b);
	modul++;
	//Convert newKB back into Montgomery space
	newKB = modul64(newKB, 0, b);
	modul++;

	if (printer) {
		end = clock();
		time_spent = (end - begin);
		printf("Cycles to do modul64 was %d\n", time_spent/2);
	}


	int maxProbeIn = 0;
	float avgProbeIn = 0;
	//int index = 0;
	for (int j = 0; j<m; j++) {

		clock_t beginindex;
		if (printer & j == m >> 1) {
			beginindex = clock();
		}

		hash = js & mem - 1;

		if (printer & j == m >> 1) {
			clock_t endindex = clock();
			time_spent = (endindex - beginindex);
			printf("Cycles to calculate hash and index was %d\n", time_spent);
		}

		//Basic linear probing
		for (int probe = 0; probe < m; probe++) {
			avgProbeIn++;
			if (probe > maxProbeIn) {
				maxProbeIn = probe;
			}
			lookups++;

			clock_t beginhash;
			if (printer & j == m >> 1) {
				beginhash = clock();
			}

			//if (hashKeys[(Sm + hash)] == 0) {
			if ((bits[S*ints + (hash / 32)] & (1 << (hash & 31))) == 0) {
				bits[S*ints + (hash / 32)] += 1 << (hash & 31);

				//Don't store it - we'll try and re-calculate it
				hashKeys[(Sm + hash)] = js; //This costs around 3750 cycles

				if (printer & j == m >> 1) {
					clock_t endhash = clock();
					time_spent = (endhash - beginhash);
					printf("Cycles to add key and value to hash table was %d\n", time_spent);
				}

				break;
			}

			hash = (hash + ((probe + 1)*(probe + 1))) & (mem - 1);

		}

		clock_t beginmul;
		if (printer & j == m >> 1) {
			beginmul = clock();
		}

		js = montmul(js, newKB, b, bprime);
		montmuls++;

		if (printer & j == m >> 1) {
			clock_t endmul = clock();
			time_spent = (endmul - beginmul);
			printf("Cycles to perform a montmul was %d\n", time_spent);
		}

	}
	if (printer) {
		printf("Number of baby steps: %d\n", m);
		printf("Number of lookups while inserting into the hash table was %d\n", lookups);
		printf("Max probe length for insert was %d\n", maxProbeIn);
		printf("Avg probe length for insert was %f\n", avgProbeIn / m);
	}


	//Finished calculating the hash table --------------------------------------------------------------------

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles calculating new hash table was %d\n", time_spent);
		printf("Average was %d\n", time_spent / m);
	}

	begin = clock();
	//Compute KernelBase^-m (mod b)
	unsigned long long c1 = newKB2;
	c1 = modul64(newKB2, 0, b);
	modul++;

	//This should be KernelBase^-1 (mod b)
	//Now repeatedly square it as m is a power of two

	for (int t = 0; t<shift; t++) {
		c1 = montmul(c1, c1, b, bprime);
		montmuls++;
	}

	long long output = -5;

	int tMin = NMin >> shift;

	if (printer) {
		printf("tMin = %d\n", tMin);
	}

	lookups = 0;
	int countmuls = tMin;
	int giant = 0;
	int collisions = 0;

	int maxProbe = 0;
	float avgProbe = 0;

	unsigned long long fixedBeta = 0;
	unsigned long long fixedsB = modul64(1, 0, b);
	modul++;
	unsigned long long beta = 0;

	bool skip = false;
	int thisk = 0;
	//The first 3 values of ks contain NMin, NMax and Q now, so start at 3.
	for (int k = 3; k < *counterIn; k++) {
		if (skip) {
			skip = false;
		}

		else if (dev_c[k] == -1) {
			//This isn't a k-value, it proves the k-value is next
			thisk = dev_c[k+1];
			fixedBeta = modul64(thisk, 0, b);
			modul++;
			skip = true;
		}
		else {
			int remainder = dev_c[k];

			//Using subsequence k should be updated to be k*b^remainder
			unsigned long long sB = fixedsB;
			for (int rem = 0; rem < remainder; rem++) {
				sB = montmul(sB, KernelBase, b, bprime);
				montmuls++;
			}

			beta = montmul(fixedBeta, sB, b, bprime);
			montmuls++;

			for (int t = 0; t < tMin; t++) {
				beta = montmul(beta, c1, b, bprime);
				montmuls++;
			}

			if (printer & k == 0) {
				printf("We're in that bit that crashes!\n");
			}

			for (int t = tMin; t < (NMax/m)+1; t++) {
				giant++;

				//Check if beta is in js
				hash = beta & mem - 1;
				//index = hash*N + S;

				//Its possible beta is here, use linear probing to check
				for (int probe = 0; probe < m; probe++) {
					avgProbe++;
					if (probe > maxProbe) {
						maxProbe = probe;
					}
					lookups++;
					//if (hashKeys[(Sm + hash)] == 0) {
					if ((bits[S*ints + (hash / 32)] & (1 << (hash & 31))) == 0) {
						//Beta is not here
						break;
					}
					collisions++;

					if (hashKeys[Sm + hash] == (int)beta) {

						lookups++;
						//printf("Match in Thread %d, Block %d. t=%d, hash=%d, probe=%d beta=%llu. Output will be %llu | %d*%d^%d-1\n", i, block, t, hash, probe, beta, b, ks[k], outputBase, output);

						//We've found beta
						//We've had a match
						//Find the j value
						unsigned long long jsnew = 1;
						jsnew = modul64(jsnew, 0, b);
						modul++;
						for (int jval = 0; jval < m; jval++) {
							if (jsnew == beta) {
								output = t*m + jval;
								break;
							}
							jsnew = montmul(jsnew, newKB, b, bprime);
							montmuls++;
						}
						//printf("Match in Thread %d, Block %d. t=%d, hash=%d, probe=%d beta=%llu. Output will be %llu | %d*%d^%d-1\n", i, block, t, hash, probe, beta, b, ks[k], outputBase, output);
						break;
					}

					hash = (hash + ((probe + 1)*(probe + 1))) & (mem - 1);
				}

				beta = montmul(beta, c1, b, bprime);
				countmuls++;
				montmuls++;
			}

			if (output < NMin) {
				output = -3;
			}
			else if (output > NMax) {
				output = -4;
			}
			else {
				printf("Output will be %llu | %d*%d^%d-1\n", b, thisk, *Base, (output*Q)+ dev_c[k]);
				output = -5;
			}
		}

	}
	if (printer) {
		printf("Number of giant steps: %d\n", giant);
		printf("Number of collisions: %d\n", collisions);
		printf("Number of lookups against hash table was %d\n", lookups);
		printf("Max probe length was %d\n", maxProbe);
		printf("Average probe length was %f\n", avgProbe / giant);
	}

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles to complete BSGS step was %d\n", time_spent);
		printf("Average (BSGS Cycles/muls) was %d\n", (time_spent / countmuls));
		printf("Average (BSGS Cycles/lookups) was %d\n", (time_spent / lookups));
	}

	begin = clock();

	NOut[S] = output; //This should contain the k-value in the top 32 bits and the n-value in the low 32 bits

	end = clock();
	time_spent = (end - begin);
	if (printer) {
		printf("Cycles to write output to NOut was %d\n", time_spent);
		printf("Total number of montgomery multiplies was %d\n", montmuls);
		printf("Total number of calls to modul64 was %d\n", modul);
		printf("Total number of calls to mulul64 was %d\n", mulul);
	}

	time_spent = (end - beginfull);
	if (printer) {
		printf("Cycles to execute one full thread was %d\n", time_spent);
	}
}


//Update this at some point to use getopt
int main(int argc, char* argv[])
{
	const int kb = 1024;
	const int mb = kb * kb;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
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

	unsigned long long targetHashSize = 1;
	while (targetHashSize < props.totalGlobalMem / 2) {
		targetHashSize = targetHashSize << 1;
	}
	wcout << "Should target a " << targetHashSize / mb << "mb hash table" << endl << endl;

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

	//Second pass through the ABCD file to write the values into the matrix
	//Store the k and n values in this array
	//boost::numeric::ublas::matrix<int> kns(count1, max);
	//std::list<int> kns;
	count3 = count3 + (2 * count1);
	int *kns = (int *)malloc(count3*sizeof(int));
	int *ks = (int *)malloc(count1*sizeof(int));

	int minN = INT_MAX;
	int maxN = 0;

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
				if (base == 0) {
					token = token.substr(n + 1);
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
	cout << "Find the optimal(ish) Q value for subsequences in base b^Q" << endl;
	int range = maxN - minN;
	int minWork = count1*range;
	int minSubs = count1;
	int minRange = range;
	int minQ = 0;
	cout << count1 << " k-values over a range of " << range << " n-values" << endl;
	cout << "Q=1 work: " << minWork << endl;
	//Iterate through Q values - from 2 to 24(?)
	for (int Q = 2; Q < 24; Q++) {
		int subsequences = 0;
		//Count the number of subsequences
		bool *subseq = (bool *)malloc(count1*Q*sizeof(bool));
		memset(subseq, false, count1*Q*sizeof(bool));
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
			minWork = subsequences*range;
			minSubs = subsequences;
			minRange = range;
		}
		cout << "Q=" << Q << ". Work= " << subsequences*range << ". Range = " << range << ". Subsequences= " << subsequences << endl;
		free(subseq);
	}

	//Successfully found Q --------------------------------------------------------------------------------------
	cout << "Min Work has Q=" << minQ << ". Work= " << minSubs*minRange << ". Range = " << minRange << ". Subsequences= " << minSubs << endl;

	//Recalculate the boolean array for our chosen Q
	bool *subseq = (bool *)malloc(count1*minQ*sizeof(bool));
	memset(subseq, false, count1*minQ*sizeof(bool));
	int whichk = -1;
	for (int qq = 0; qq < count3; qq++) {
		if (kns[qq] == 0) {
			whichk++;
			//Skip the k-value itself which is next in the array
			qq++;
		}
		else {
			//This is an n-value
			int mod = kns[qq] % minQ;
			subseq[whichk*minQ + mod] = true;
		}
	}

	//Edit the list of k-values to now contain the k and its associated modulo values based on Q
	//Use ks2 for now to store nMin and nMax too. 
	int *ks2 = (int *)malloc((count1*2 + minSubs + 3)*sizeof(int));
	ks2[0] = minN;
	ks2[1] = maxN;
	ks2[2] = minQ;
	int ks2counter = 3;
	for (int kvalues = 0; kvalues < count1; kvalues++) {
		//Seperate k values with a -1 marker
		ks2[ks2counter] = -1;
		ks2counter++;

		ks2[ks2counter] = ks[kvalues];
		ks2counter++;
		for (int Q = 0; Q < minQ; Q++) {
			if (subseq[kvalues*minQ + Q] == true) {
				ks2[ks2counter] = Q;
				ks2counter++;
			}
		}
	}

	free(subseq);


	//Generate Primes -------------------------------------------------------------------------------------------

	int blockScale = 32; //Default scaling
	int threadScale = 2; //Default scaling 
	//Use the input arguments to change blocks and threads. 
	if (argc == 2) {
		//Assume only a threadScale - we must check this is a power of 2 at some point
		threadScale = atoi(argv[1]);
	}
	if (argc == 3) {
		//Assume threadScale and then blockScale - - we must check these are both a power of 2 at some point
		threadScale = atoi(argv[1]);
		blockScale = atoi(argv[2]);
	}

	const int blocks = 32 * blockScale;
	const int threads = 32 * threadScale; //These must multiply to around 65536. Larger and CUDA times out
	const int arraySize = blocks*threads;
	const int testArraySize = arraySize * 24;
	const int hashScaling = 4;

	//Use targetHashSize to set up the hash table - int = 32 bits = 4 bytes, so divide by 4
	//Each thread requires the a hash table, so also divide by arraySize
	long long longhashTableSize = (((targetHashSize / 4) / arraySize)/hashScaling);
	int hashTableSize = longhashTableSize;
	cout << "Each thread should have " << hashTableSize*hashScaling << " buckets, to store " << hashTableSize << " elements. (Density 1/" << hashScaling << ")" << endl;

	unsigned long long *KernelP = (unsigned long long *)malloc(arraySize*sizeof(unsigned long long));
	int *NOut = (int *)malloc(arraySize*sizeof(int));
	int *hashKeys = (int *)malloc(arraySize * hashTableSize * hashScaling * sizeof(int));
	unsigned int *bits = (unsigned int *)malloc(((arraySize * hashTableSize * hashScaling)/32) * sizeof(int));
	memset(hashKeys, 0, arraySize * hashTableSize * hashScaling * sizeof(int));
	memset(bits, 0, ((arraySize * hashTableSize * hashScaling)/32) * sizeof(int));


	//Low should be greater than the primes we use below. 
	//unsigned long long low = 6000000000;
	//unsigned long long high = 6004000000;

	//unsigned long long low = 1000067500000;
	//unsigned long long high = 1000070000000;

	//unsigned long long low = 1000099000000;
	//unsigned long long high = 1000100000000;

	//unsigned long long low = 102254819500000L;
	unsigned long long low = 102297149770000L;
	unsigned long long high = 102297160000000L;

	//unsigned long long low = 600000;
	//unsigned long long high = 10000000;


	unsigned long long startLow = low; //Don't touch this. Used for timing purposes

	//Use the idea of a segmented sieve. Generate a list of small primes first
	//Could use the first 1024 primes as a starter. 8161 is the 1024th prime

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
			for (int i = p * 2; i < smallPrimes; i += p) {
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

	//cudaStatus = cudaMalloc((void**)&dev_c, (count1 * 2 + minSubs + 3) * sizeof(int));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

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

	cudaStatus = cudaMalloc((void**)&dev_j, ((arraySize * hashTableSize * hashScaling) / 32) * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Copy the data to the correct GPU buffers

	//Lets try storing the k values and remainders in constant memory instead
	cudaStatus = cudaMemcpyToSymbol(dev_c, ks2, (count1 * 2 + minSubs + 3) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to constant memory failed!");
		cout << (count1 * 2 + minSubs + 3) * sizeof(int) << "bytes" << endl;
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_e, &base, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_f, &ks2counter, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_g, hashKeys, arraySize * hashTableSize * hashScaling * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
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

	cudaStatus = cudaMemcpy(dev_j, bits, ((arraySize * hashTableSize * hashScaling) / 32) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
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
		cudaError_t cudaStatus = addWithCuda(NOut, KernelP, arraySize, blocks, threads, hashKeys, hashTableSize, hashScaling, bits);
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
	cout << "Progress = " << KernelP[arraySize - 1] - startLow << " at " << (KernelP[arraySize - 1] - startLow) / time_spent << " p/sec" << endl << endl;

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

	cout << "Each thread used " << hashTableSize*hashScaling << " buckets, to store " << hashTableSize << " elements. (Density 1/" << hashScaling << ")" << endl;
	cout << "Hash table size was " << (hashTableSize*hashScaling * 4 * arraySize) / mb << "mb of GPU RAM" << endl;
	cout << "Blocksize = " << blocks << ". Threads per block = " << threads << "." << endl;

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_e);
	cudaFree(dev_f);
	cudaFree(dev_g);

	cudaFree(dev_j);
	return cudaStatus;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *NOut, unsigned long long *KernelP, unsigned int size, const int blocks, const int threads, int *hashKeys, int hashElements, int hashDensity, unsigned int *bits)
{

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_b, KernelP, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
	}

	cudaStatus = cudaMemset(dev_g, 0, size * hashElements * hashDensity * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy input failed!");
	}

	cudaStatus = cudaMemset(dev_j, 0, ((size * hashElements * hashDensity) / 32) * sizeof(unsigned int));
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
	addKernel1 <<<blocks, threads, 0, stream0 >>>(dev_a, dev_b, /*dev_c, */dev_e, dev_f, dev_g, dev_h, dev_i, dev_j);
	
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

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	// Copy output vector from GPU buffer to host memory.
	//cudaStatus = cudaMemcpy(NOut, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy output failed!");
	//}

	return cudaStatus;
}

