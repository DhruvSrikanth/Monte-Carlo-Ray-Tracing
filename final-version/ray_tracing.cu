#include <iostream>
using namespace std;
#include <chrono>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <random>
#include<cuda.h>
#include<cuda_runtime.h>

#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a,b) (((a)<(b))?(a):(b))

struct Point {
    double x;
    double y;
    double z;
};

void print_point(Point p) {
    // Print the point
    cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
}

__global__ Point vec_add(Point p1, Point p2) {
    // Add two vectors
    Point res;
    res.x = p1.x + p2.x;
    res.y = p1.y + p2.y;
    res.z = p1.z + p2.z;
    return res;
}

__global__ Point vec_scale(Point p, double scale) {
    // Scale a vector
    Point res;
    res.x = p.x * scale;
    res.y = p.y * scale;
    res.z = p.z * scale;
    return res;
}

__global__ double vec_dotp(Point p1, Point p2) {
    // Dot product of two vectors
    double res;
    res = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
    return res;
}

__global__ Point vec_direction(Point p1, Point p2) {
    // Direction of p2 from p1
    Point res;
    double magnitude = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
    res.x = (p2.x - p1.x) / magnitude;
    res.y = (p2.y - p1.y) / magnitude;
    res.z = (p2.z - p1.z) / magnitude;
    return res;
}

// TODO: Change to fast forwarding LCG PRNG
__global__ Point direction_sampling() {
    Point V;
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr_psi(0, 2*M_PI);
    uniform_real_distribution<double> distr_cos_theta(-1.0, 1.0);

    double phi = distr_psi(eng);
    double cos_theta = distr_cos_theta(eng);
    double sin_theta = sqrt(1 - pow(cos_theta, 2));
    V.x = sin_theta * cos(phi);
    V.y = sin_theta * sin(phi);
    V.z = cos_theta;
    return V;
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

__global__ void ray_tracing(double* grid, int N_rays, int N_gridpoints) {
    // Initialize points
    Point W, V, I, N, S;

    int per_point_threads = 3;
    int per_point_blocks = 1;

    // Initialize simulation parameters
    double w_max = 10.0;
    Point L = {4,4,-1};
    Point C = {0,12,0};
    double r = 6.0;
    double Wy = 10.0;

    double t;
    double b;

    // CUDA timer
    cudaEvent_t start_device, stop_device;  
    double time_device;

    // Create timers
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    // Index variables
    int starting_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int n = starting_index; n < N_rays; n+=stride) {

        // Start timer
        cudaEventRecord(start_device, 0);  

        while (true) {
            // sample random v from unit sphere
            V = direction_sampling<<<per_point_blocks, per_point_threads>>>();
            W = vec_scale<<<per_point_blocks, per_point_threads>>>(V, Wy / V.y);
            bool condition = abs(W.x) < w_max && abs(W.z) < w_max && pow(vec_dotp<<<per_point_blocks, per_point_threads>>>(V, C), 2) + r*r - vec_dotp<<<per_point_blocks, per_point_threads>>>(C, C) > 0;
            if (condition) {
                break;
            }
        }

        t = vec_dotp<<<per_point_blocks, per_point_threads>>>(V,C) - sqrt(pow(vec_dotp<<<per_point_blocks, per_point_threads>>>(V,C), 2) + r*r - vec_dotp<<<per_point_blocks, per_point_threads>>>(C, C));
        I = vec_scale<<<per_point_blocks, per_point_threads>>>(V, t);
        N = vec_direction<<<per_point_blocks, per_point_threads>>>(C, I);
        S = vec_direction<<<per_point_blocks, per_point_threads>>>(I, L);
        b = max(0.0, vec_dotp<<<per_point_blocks, per_point_threads>>>(S, N));

        // Compute the grid point indices
        int i = (W.z + w_max) / (2*w_max) * N_gridpoints;
        int j = (W.x + w_max) / (2*w_max) * N_gridpoints;
        int idx_1d_local = i * N_gridpoints + j;
        // Update the grid point
        // TODO: make this thread safe
        grid[idx_1d_local] += b;

        // Stop timer
        cudaEventRecord(stop_device, 0);
        cudaEventSynchronize(stop_device);
        cudaEventElapsedTime(&time_device, start_device, stop_device);
        cout << "Iteration: " << n << " - Grind Rate: " << floor(1e3/time_device) << " rays/sec" << endl;
    }

    // Release the memory for the timer
    cudaEventDestroy(start_device);
    cudaEventDestroy(stop_device);
}

int main(int argc, char** argv) {
    // Initialize variables
    int N_rays = stod(argv[1]);
    int N_gridpoints = stoi(argv[2]);
    int n_threads_per_block = stoi(argv[3]);

    int n_blocks = MIN(N_gridpoints*N_gridpoints/nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);

    // CUDA timer
    cudaEvent_t start_device, stop_device;  
    double time_device;

    cout << "Simulation Parameters:" << endl;
    cout << "Number of rays = " << N_rays << endl;
    cout << "Number of gridpoints = " << N_gridpoints << endl;

    // Simple checks
    assert(N_rays > 0);
    assert(N_gridpoints > 0);

    // Initialize grid
    double* grid = new double[N_gridpoints*N_gridpoints];
    create_contiguous_2d_array(grid, N_gridpoints*N_gridpoints);

    // Allocate memory to the grid on the device
    double* grid_device;
    cudaMalloc(&grid_device, N_gridpoints*N_gridpoints*sizeof(double));

    // Copy the grid to the device
    cudaMemcpy(grid_device, grid, N_gridpoints*N_gridpoints*sizeof(double), cudaMemcpyHostToDevice);
    
    // Perform simulation
    ray_tracing<<<n_blocks, n_threads_per_block>>>(grid_device, N_rays, N_gridpoints);
    
    // Copy the grid from the device to the host
    cudaMemcpy(grid, grid_device, N_gridpoints*N_gridpoints*sizeof(double), cudaMemcpyDeviceToHost);

    // Write to file
    write_to_file(grid, "./output/output.txt", N_gridpoints*N_gridpoints, N_rays-1);

    // Release the memory for the grid
    cudaFree(grid_device); // Free the memory for the grid on the device
    delete[] grid; // Free the memory for the grid on the host

    return 0;

}
