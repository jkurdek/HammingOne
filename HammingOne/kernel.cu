#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <chrono>

constexpr auto vector_count = 100000;
constexpr auto vector_length = 1000;
constexpr auto prime = 1900813U;
constexpr auto fnv_prime = 16777619U;
constexpr auto bucket_size = 512;
constexpr auto table_size = 192;
constexpr auto max_iterations = 25;
constexpr auto preferred_bucket_size = 409;

using namespace std;

cudaError_t compute_pairs_with_cuda(const char* data, const int no_of_buckets, uint64_t* powers, int* seeds);


//invoke correct bucket hash function based on parameter function. (Each table within a bucket has its own hash fucntion h1, h2, h3)
__device__ int bucket_hash_function(const int function, const int seed, const uint64_t hs)
{
	const auto c1 = 13363817 ^ seed, c2 = 30262931 ^ seed, c3 = 29317751 ^ seed, c4 = 6081871 ^ seed, c5 = 12156847 ^ seed, c6 = 14096701 ^ seed;
	switch (function)
	{
	case 1:
		return ((c1 + c2 * hs) % prime) % table_size;
	case 2:
		return ((c3 + c4 * hs) % prime) % table_size;
	case 3:
		return ((c5 + c6 * hs) % prime) % table_size;
	}
	return -1;
}

__host__ void compute_power_primes(uint64_t* powers)
{
	powers[0] = 1;
	for (auto i = 1; i <= vector_length; i++)
	{
		powers[i] = powers[i - 1] * fnv_prime;
	}
}

//calculate main hash for a vector. Use precomputed powers for fast computation.
__device__ uint64_t main_hash(const char* vector, const uint64_t* powers)
{
	uint64_t hash = 0;
	for (auto i = 0; i < vector_length; i++)
	{
		hash += vector[i] * powers[vector_length - i];
	}

	return hash;
}

__device__ int assign_bucket(const uint64_t hash, const int no_of_buckets)
{
	return hash % no_of_buckets;
}

//main algorithm: for each vector generate all possible hashes and check if they are present in the hash map.
__global__ void count_pairs_kernel(char* dev_data, uint64_t* dev_hash_map, uint64_t* dev_hashes, uint64_t* dev_powers, int* dev_pairs, int no_of_buckets, int* buckets_seeds)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < vector_count)
	{
		uint64_t mg = dev_hashes[i];
		for (int j = 0; j < vector_length; j++)
		{
			auto mh = mg;
			if (*(dev_data + i * vector_length + j) == '0') mh += dev_powers[vector_length - j];
			else mh -= dev_powers[vector_length - j];



			auto b = assign_bucket(mh, no_of_buckets);
			const auto h1 = bucket_hash_function(1, buckets_seeds[b], mh);

			if (dev_hash_map[table_size * 3 * b + h1] == mh)
			{
				atomicAdd(dev_pairs + i, 1);
				continue;
			}

			const auto h2 = bucket_hash_function(2, buckets_seeds[b], mh);
			if (dev_hash_map[table_size * 3 * b + table_size + h2] == mh)
			{
				atomicAdd(dev_pairs + i, 1);
				continue;
			}

			const auto h3 = bucket_hash_function(3, buckets_seeds[b], mh);
			if (dev_hash_map[table_size * 3 * b + 2 * table_size + h3] == mh)
			{
				atomicAdd(dev_pairs + i, 1);
			}

		}

	}
}

/* bucket hashing. Each bucket tries iteratively to put all the keys in its tables.
The operation is reapeated until all keys are placed.
If the iteration limit is reached we conclude that we chose a wrong hashing function, we generate new random seed and try again */
__global__ void hash_map_construction_kernel(uint64_t* dev_hash_map, uint64_t* dev_buffer, int* dev_bucket_count, int* dev_seeds, int* bucket_seeds)
{
	int x = 0;
	const int bucket_no = blockIdx.x;
	const int global_no = dev_bucket_count[bucket_no] + threadIdx.x;
	const int threadNo = threadIdx.x;


	int attempt = 0;

	__shared__  uint64_t table1[table_size];
	__shared__  uint64_t table2[table_size];
	__shared__  uint64_t table3[table_size];

	int placed = 1;


	if (global_no < dev_bucket_count[bucket_no + 1])
	{


		do
		{
			placed = 0;
			if (threadNo < table_size)
			{
				table1[threadNo] = 0;
				table2[threadNo] = 0;
				table3[threadNo] = 0;
			}

			int seed = dev_seeds[(blockIdx.x * 6 + attempt)];

			if (threadNo != 0)
			{
				bucket_seeds[bucket_no] = seed;
			}

			int hs1 = bucket_hash_function(1, seed, dev_buffer[global_no]);
			int hs2 = bucket_hash_function(2, seed, dev_buffer[global_no]);
			int hs3 = bucket_hash_function(3, seed, dev_buffer[global_no]);

			for (auto k = 0; k < max_iterations; k++)
			{
				const auto t = k % 3 + 1;

				auto* tx = table1;
				auto hs = hs1;
				if (t == 1)
				{
					tx = table1;
					hs = hs1;
				}
				else if (t == 2)
				{
					tx = table2;
					hs = hs2;
				}
				else if (t == 3)
				{
					tx = table3;
					hs = hs3;
				}


				if (placed == 0)
				{
					tx[hs] = dev_buffer[global_no];
				}

				//we need to sync threads before checking wherther the key is placed
				if (__syncthreads_and(placed == 1)) break;

				if (table1[hs1] == dev_buffer[global_no] || table2[hs2] == dev_buffer[global_no] || table3[hs3] == dev_buffer[global_no]) placed = 1;
				else placed = 0;


			}
			attempt++;

			x = __syncthreads_and(placed == 1);

		} while (x == 0);

	}

	//copying data into correct places in global memory
	if (threadNo < table_size)
	{
		dev_hash_map[table_size * 3 * (bucket_no)+threadNo] = table1[threadNo];
		dev_hash_map[table_size * 3 * (bucket_no)+table_size + threadNo] = table2[threadNo];
		dev_hash_map[table_size * 3 * (bucket_no)+2 * table_size + threadNo] = table3[threadNo];
	}



}
__global__ void buffer_construction_kernel(uint64_t* dev_hashes, uint64_t* dev_buffer, int* dev_bucket_count, int* dev_bucket_offset, const int no_of_buckets)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < vector_count)
	{
		const auto hash = dev_hashes[i];
		dev_buffer[dev_bucket_count[assign_bucket(hash, no_of_buckets)] + dev_bucket_offset[i]] = hash;
	}
}

__global__ void bucket_assignement_kernel(const char* dev_data, uint64_t* dev_hashes, int* dev_bucket_count, int* dev_bucket_offset, const int no_of_buckets, uint64_t* dev_powers)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < vector_count)
	{
		uint64_t hs = main_hash(dev_data + i * vector_length, dev_powers);
		dev_hashes[i] = hs;

		int bucketNo = assign_bucket(hs, no_of_buckets);
		dev_bucket_offset[i] = atomicAdd(dev_bucket_count + bucketNo, 1);
	}
}

int main()
{
	//Fast IO initialization
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	//allocation of input table and data reading
	char* data = new char[vector_count * vector_length];

	ifstream is("input.txt");

	char v;
	int max = vector_count * vector_length;
	for (int i = 0; i < max; i++)
	{
		is >> v;
		data[i] = v;
	}

	const auto start = std::chrono::high_resolution_clock::now();

	//Precompute powers of fnv_prime for fast hashing
	uint64_t powers[vector_length + 1];
	compute_power_primes(powers);

	//calculate amount of buckets. We want to get 80% occupancy on average to prevent bucket overflow
	const auto no_of_buckets = (vector_count + preferred_bucket_size - 1) / preferred_bucket_size;

	//generate a set of random numbers to seed GPU hashing functions
	int* seeds = new int[no_of_buckets * 6];

	for (int i = 0; i < no_of_buckets * 6; i++)
	{
		seeds[i] = rand() * rand();
	}

	// Compute number of pairs
	cudaError_t cudaStatus = compute_pairs_with_cuda(data, no_of_buckets, powers, seeds);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "compute_pairs_with_cuda failed!");
		return 1;
	}


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	const auto finish = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	delete[] data;
	delete[] seeds;

	return 0;
}

// Helper function for using CUDA to calculate pairs with Hamming distance 1
cudaError_t compute_pairs_with_cuda(const char* data, const int no_of_buckets, uint64_t* powers, int* seeds)
{

	//declaration of pointers
	char* dev_data = nullptr;
	uint64_t* dev_buffer = nullptr;
	uint64_t* dev_hashes = nullptr;
	uint64_t* dev_powers = nullptr;
	int* dev_bucket_count = nullptr;
	int* dev_bucket_offset = nullptr;
	uint64_t* dev_hash_map = nullptr;
	int* dev_seeds = nullptr;
	int* dev_bucket_seeds = nullptr;
	int* dev_pairs = nullptr;

	cudaEvent_t start, stop;
	float time = 0, time_temp;
	cudaError_t cudaStatus;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);



#pragma region mem_alloc
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_data, vector_length * vector_count * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: data");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_buffer, vector_count * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: buffer");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_hashes, vector_count * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: hashes");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bucket_count, (no_of_buckets + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: bucket_count");
		goto Error;
	}

	cudaStatus = cudaMemset(dev_bucket_count, 0, (no_of_buckets + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!: bucket_count");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bucket_offset, vector_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: bucket_offset");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_hash_map, 3 * table_size * no_of_buckets * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: hash_map");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_powers, (vector_length + 1) * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: powers");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_seeds, (no_of_buckets * 6) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: seeds");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bucket_seeds, (no_of_buckets) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: bucket_seeds");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pairs, (vector_count) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: pairs");
		goto Error;
	}

	cudaStatus = cudaMemset(dev_pairs, 0, (vector_count) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!: pairs");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_data, data, vector_length * vector_count * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: data");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_powers, powers, (vector_length + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: powers");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_seeds, seeds, (no_of_buckets * 6) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed: seeds");
		goto Error;
	}
#pragma endregion

	int threads_per_block = 512;
	int no_of_blocks = vector_count + threads_per_block - 1 / threads_per_block;

	cudaEventRecord(start, 0);

	// Assign bucket for each vector
	bucket_assignement_kernel << <no_of_blocks, threads_per_block >> > (dev_data, dev_hashes, dev_bucket_count, dev_bucket_offset, no_of_buckets, dev_powers);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_temp, start, stop);
	time += time_temp;

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bucket_assignement_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bucket_assignement_kernel!\n", cudaStatus);
		goto Error;
	}



	//calculate buckets start points in buffer array
	thrust::exclusive_scan(thrust::device, dev_bucket_count, dev_bucket_count + no_of_buckets + 1, dev_bucket_count, 0);


	cudaEventRecord(start, 0);

	//construct buffer table from data
	buffer_construction_kernel << < no_of_blocks, threads_per_block >> > (dev_hashes, dev_buffer, dev_bucket_count, dev_bucket_offset, no_of_buckets);

	hash_map_construction_kernel << < no_of_buckets, bucket_size >> > (dev_hash_map, dev_buffer, dev_bucket_count, dev_seeds, dev_bucket_seeds);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_temp, start, stop);
	time += time_temp;

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "buffer_construction_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching buffer_construction_kernel!\n", cudaStatus);
		goto Error;
	}


	cudaEventRecord(start, 0);

	//construct hash_table from buffer
	hash_map_construction_kernel << < no_of_buckets, bucket_size >> > (dev_hash_map, dev_buffer, dev_bucket_count, dev_seeds, dev_bucket_seeds);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_temp, start, stop);
	time += time_temp;

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "hash_map_construction_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching hash_map_construction_kernel!\n", cudaStatus);
		goto Error;
	}


	cudaEventRecord(start, 0);
	//count no of hamming pairs

	count_pairs_kernel << < no_of_blocks, threads_per_block >> > (dev_data, dev_hash_map, dev_hashes, dev_powers, dev_pairs, no_of_buckets, dev_bucket_seeds);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_temp, start, stop);
	time += time_temp;

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "hash_map_construction_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching hash_map_construction_kernel!\n", cudaStatus);
		goto Error;
	}



	int result = thrust::reduce(thrust::device, dev_pairs, dev_pairs + vector_count, 0);
	cout << "Sum of kernels execution time [ms]: " << time << endl;
	cout << "No of pairs: "<<  result << endl;


Error:
	cudaFree(dev_data);
	cudaFree(dev_buffer);
	cudaFree(dev_bucket_count);
	cudaFree(dev_bucket_offset);
	cudaFree(dev_hashes);
	cudaFree(dev_powers);
	cudaFree(dev_hash_map);
	cudaFree(dev_seeds);
	cudaFree(dev_bucket_seeds);
	cudaFree(dev_pairs);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return cudaStatus;
}


