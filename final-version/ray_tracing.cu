#include <iostream>
using namespace std;
#include <assert.h>
#include <math.h>
#include <fstream>
#include <random>
#include<cuda.h>
#include<cuda_runtime.h>

// Maximum number of blocks in a device grid (for dim x)
#define MAX_BLOCKS 2147483647

// Min function
#define MIN(a,b) (((a)<(b))?(a):(b))

// Data structure for a point
struct Point {
    // x, y, z coordinates
    double x;
    double y;
    double z;
};

// HOST FUNCTIONS

void print_point(Point p) {
    // Print the point
    cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
}

void write_to_file(double* output, string filename, int N, int NT) { 
    // Allocate memory for the file
    ofstream file;
    file.open(filename);

    // Write the timestep to the file
    file << "[" << NT << "], ";

    // Write the data to the file
    file << "[";
    int n = sqrt(N);
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            file << "[";
        } 
        else {
            file << ", [";
        }
        for (int j = 0; j < n; j++) {
            int idx_1d = i * n + j;
            if (j == N - 1) {
                file << output[idx_1d];
            } 
            else {
                file << output[idx_1d] << ", ";
            }
        }
        file << "]";
    }
    file << "]";

    // Release the memory for the file
    file.close();
}

void create_contiguous_2d_array(double* mat, int N) {
    // Create a contiguous 2d array
    int n = sqrt(N);
    // Initialize the matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx_1d = i*n +j;
            mat[idx_1d] = 0.0;
        }
    }
}

// DEVICE FUNCTIONS

__device__ Point vec_add(Point p1, Point p2) {
    // Add two vectors
    Point res;
    res.x = p1.x + p2.x;
    res.y = p1.y + p2.y;
    res.z = p1.z + p2.z;
    return res;
}

__device__ Point vec_scale(Point p, double scale) {
    // Scale a vector
    Point res;
    res.x = p.x * scale;
    res.y = p.y * scale;
    res.z = p.z * scale;
    return res;
}

__device__ double vec_dotp(Point p1, Point p2) {
    // Dot product of two vectors
    double res;
    res = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
    return res;
}

__device__ Point vec_direction(Point p1, Point p2) {
    // Direction of p2 from p1
    Point res;
    double magnitude = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
    res.x = (p2.x - p1.x) / magnitude;
    res.y = (p2.y - p1.y) / magnitude;
    res.z = (p2.z - p1.z) / magnitude;
    return res;
}

__device__ double LCG_random_double(uint64_t& seed) {
    // Generate a random number between 0 and 1 from a uniform distribution
    const uint64_t m = 9223372036854775808ULL; // 2ˆ63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    seed = (a * seed + c ) % m;
    return (double) (seed) / (double) m;
}

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
    // Fast forward the LCG by n steps
    const uint64_t m = 9223372036854775808ULL; // 2ˆ63
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;
    while (n >0) {
        if (n & 1) {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= ( a + 1);
        a *= a;
        n >>= 1;
    }
    return (a_new * seed + c_new) % m;
}

__device__ Point direction_sampling(uint64_t& seed) {
    // Generate a random ray
    Point V;

    // Sample point for respective distributions
    double phi = 2*M_PI*LCG_random_double(seed); // 0 ~ 2*pi
    double cos_theta = 2*LCG_random_double(seed) - 1; // -1 ~ 1
    double sin_theta = sqrt(1 - pow(cos_theta, 2));
    
    V.x = sin_theta * cos(phi);
    V.y = sin_theta * sin(phi);
    V.z = cos_theta;

    return V;
}

__global__ void ray_tracing(double* grid, int* N_gridpoints) {
    // Initialize points
    Point W, V, I, N, S;

    // Initialize simulation parameters
    double w_max = 10.0;
    Point L = {4,4,-1};
    Point C = {0,12,0};
    double r = 6.0;
    double Wy = 10.0;

    // Initialize variables
    double t;
    double b;

    uint64_t init_seed = 1;
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = 200 * tid;

    while (true) {
        // Generate random seed
        uint64_t seed = fast_forward_LCG(init_seed, n);
        
        // sample random v from unit sphere
        V = direction_sampling(seed);
        W = vec_scale(V, Wy / V.y);
        bool condition = abs(W.x) < w_max && abs(W.z) < w_max && pow(vec_dotp(V, C), 2) + r*r - vec_dotp(C, C) > 0;
        if (condition) {
            break;
        }
        init_seed = seed;
    }

    t = vec_dotp(V,C) - sqrt(pow(vec_dotp(V,C), 2) + r*r - vec_dotp(C, C));
    I = vec_scale(V, t);
    N = vec_direction(C, I);
    S = vec_direction(I, L);
    b = max(0.0, vec_dotp(S, N));

    // Compute the grid point indices
    int i = (W.z + w_max) / (2*w_max) * (*N_gridpoints);
    int j = (W.x + w_max) / (2*w_max) * (*N_gridpoints);
    int idx_1d_local = i * (*N_gridpoints) + j;

    // Update the grid point
    atomicAdd(&grid[idx_1d_local], b);
    // grid[idx_1d_local] += b; // For the serial version
}

int main(int argc, char** argv) {
    // Initialize variables
    int N_rays = stod(argv[1]);
    int N_gridpoints = stoi(argv[2]);
    int n_threads_per_block = stoi(argv[3]);

    // Compute the number of blocks
    int n_blocks = MIN(N_rays/n_threads_per_block + 1, MAX_BLOCKS);

    cout << "Simulation Parameters:" << endl;
    cout << "Number of rays = " << N_rays << endl;
    cout << "Number of gridpoints = " << N_gridpoints << endl;

    // Simple checks
    assert(N_rays > 0);
    assert(N_gridpoints > 0);

    // Initialize grid
    double* grid = new double[N_gridpoints*N_gridpoints];
    create_contiguous_2d_array(grid, N_gridpoints*N_gridpoints);

    // Allocate memory for the number of grid points and the grid
    double* grid_device;
    int* N_gridpoints_device;

    cudaMalloc(&grid_device, N_gridpoints*N_gridpoints*sizeof(double));
    cudaMalloc(&N_gridpoints_device, sizeof(int));

    // Copy the params to the device
    cudaMemcpy(grid_device, grid, N_gridpoints*N_gridpoints*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(N_gridpoints_device, &N_gridpoints, sizeof(int), cudaMemcpyHostToDevice);

    // CUDA timer
    cudaEvent_t start_device, stop_device;  
    float time_device;

    // Create timers
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    // Start timer
    cudaEventRecord(start_device, 0);  
    
    // Perform simulation
    ray_tracing<<<n_blocks, n_threads_per_block>>>(grid_device, N_gridpoints_device);

    // Stop timer
    cudaEventRecord(stop_device, 0);
    cudaEventSynchronize(stop_device);
    cudaEventElapsedTime(&time_device, start_device, stop_device);

    // Print out metrics
    cout << "Grind Rate: " << N_rays / floor(1e3/time_device) << " rays/sec" << endl; 
    cout << "Time: " << time_device << " ms" << endl;
    
    // Copy the grid from the device to the host
    cudaMemcpy(grid, grid_device, N_gridpoints*N_gridpoints*sizeof(double), cudaMemcpyDeviceToHost);

    // Release the memory for the timer
    cudaEventDestroy(start_device);
    cudaEventDestroy(stop_device);

    // Write to file
    write_to_file(grid, "./output/output.txt", N_gridpoints*N_gridpoints, N_rays-1);

    // Release the memory for the grid and the number of grid points
    cudaFree(grid_device);
    cudaFree(N_gridpoints_device);
    delete[] grid;

    return 0;

}