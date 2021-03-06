#include <iostream>
using namespace std;
#include <chrono>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <random>


struct Point {
    double x;
    double y;
    double z;
};

void print_point(Point p) {
    // Print the point
    cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
}

Point vec_add(Point p1, Point p2) {
    // Add two vectors
    Point res;
    res.x = p1.x + p2.x;
    res.y = p1.y + p2.y;
    res.z = p1.z + p2.z;
    return res;
}

Point vec_scale(Point p, double scale) {
    // Scale a vector
    Point res;
    res.x = p.x * scale;
    res.y = p.y * scale;
    res.z = p.z * scale;
    return res;
}

double vec_dotp(Point p1, Point p2) {
    // Dot product of two vectors
    double res;
    res = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
    return res;
}

Point vec_direction(Point p1, Point p2) {
    // Direction of p2 from p1
    Point res;
    double magnitude = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
    res.x = (p2.x - p1.x) / magnitude;
    res.y = (p2.y - p1.y) / magnitude;
    res.z = (p2.z - p1.z) / magnitude;
    return res;
}

Point direction_sampling() {
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

void ray_tracing(double* grid, int N_rays, int N_gridpoints) {

    // Initialize rays
    Point W, V, I, N, S;

    // Initialize simulation parameters
    double w_max = 10.0;
    Point L = {4,4,-1};
    Point C = {0,12,0};
    double r = 6.0;
    double Wy = 10.0;

    double t;
    double b;

    for (int n = 0; n < N_rays; n++) {

        auto t1 = std::chrono::high_resolution_clock::now();

        while (true) {
            // sample random v from unit sphere
            V = direction_sampling();
            W = vec_scale(V, Wy / V.y);
            bool condition = abs(W.x) < w_max && abs(W.z) < w_max && pow(vec_dotp(V, C), 2) + r*r - vec_dotp(C, C) > 0;
            if (condition) {
                break;
            }
        }

        t = vec_dotp(V,C) - sqrt(pow(vec_dotp(V,C), 2) + r*r - vec_dotp(C, C));
        I = vec_scale(V, t);
        N = vec_direction(C, I);
        S = vec_direction(I, L);
        b = max(0.0, vec_dotp(S, N));

        // Compute the grid point
        int i = (W.z + w_max) / (2*w_max) * N_gridpoints;
        int j = (W.x + w_max) / (2*w_max) * N_gridpoints;
        int idx_1d = i * N_gridpoints + j;    
        grid[idx_1d] += b;

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        cout << "Iteration: " << n << " - Grind Rate: " << floor(1/(1e-6*duration.count())) << " rays/sec" << endl;
    }
}

int main(int argc, char** argv) {
    // Initialize variables
    int N_rays = stod(argv[1]);
    int N_gridpoints = stoi(argv[2]);

    cout << "Simulation Parameters:" << endl;
    cout << "Number of rays = " << N_rays << endl;
    cout << "Number of gridpoints = " << N_gridpoints << endl;

    // Simple checks
    assert(N_rays > 0);
    assert(N_gridpoints > 0);

    // Initialize grid
    double* grid = new double[N_gridpoints*N_gridpoints];
    create_contiguous_2d_array(grid, N_gridpoints*N_gridpoints);

    // Perform simulation
    ray_tracing(grid, N_rays, N_gridpoints);

    // Write to file
    write_to_file(grid, "./output/output.txt", N_gridpoints*N_gridpoints, N_rays-1);

    // Release memory
    delete[] grid;

    return 0;
}
