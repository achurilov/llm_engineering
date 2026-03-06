
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <future>
#include <numeric>

// This function calculates the sum for a given range of iterations.
// It is designed to be highly optimized and run in a separate thread.
// It uses an algebraic simplification of the original formula to reduce the number of expensive division operations.
// Original term: 1/(i*p1+p2) - 1/(i*p1-p2)
// Simplified term: -2*p2 / ((i*p1)^2 - p2^2)
double calculate_chunk(long long start_iter, long long end_iter, int param1, int param2) {
    // Pre-calculate constants to avoid repeated calculations inside the tight loop.
    const double p1 = static_cast<double>(param1);
    const double p2 = static_cast<double>(param2);
    const double p2_sq = p2 * p2;
    const double neg_2p2 = -2.0 * p2;

    // Use multiple accumulators to break the dependency chain on a single sum variable.
    // This helps the CPU's out-of-order execution and enables compiler vectorization (SIMD).
    // We choose 8 accumulators, a good number for AVX-512 (which handles 8 doubles).
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;

    long long i = start_iter;
    const long long unroll_limit = end_iter - 7;

    // Main loop, manually unrolled 8 times for maximum instruction-level parallelism.
    for (; i <= unroll_limit; i += 8) {
        const double i0 = static_cast<double>(i);
        const double i1 = static_cast<double>(i + 1);
        const double i2 = static_cast<double>(i + 2);
        const double i3 = static_cast<double>(i + 3);
        const double i4 = static_cast<double>(i + 4);
        const double i5 = static_cast<double>(i + 5);
        const double i6 = static_cast<double>(i + 6);
        const double i7 = static_cast<double>(i + 7);
        
        const double ip0 = i0 * p1;
        const double ip1 = i1 * p1;
        const double ip2 = i2 * p1;
        const double ip3 = i3 * p1;
        const double ip4 = i4 * p1;
        const double ip5 = i5 * p1;
        const double ip6 = i6 * p1;
        const double ip7 = i7 * p1;
        
        s0 += neg_2p2 / (ip0 * ip0 - p2_sq);
        s1 += neg_2p2 / (ip1 * ip1 - p2_sq);
        s2 += neg_2p2 / (ip2 * ip2 - p2_sq);
        s3 += neg_2p2 / (ip3 * ip3 - p2_sq);
        s4 += neg_2p2 / (ip4 * ip4 - p2_sq);
        s5 += neg_2p2 / (ip5 * ip5 - p2_sq);
        s6 += neg_2p2 / (ip6 * ip6 - p2_sq);
        s7 += neg_2p2 / (ip7 * ip7 - p2_sq);
    }
    
    // Combine the partial sums from unrolling.
    double local_sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;

    // Handle any remaining iterations that didn't fit into the unrolled loop.
    for (; i <= end_iter; ++i) {
        const double current_i = static_cast<double>(i);
        const double ip = current_i * p1;
        local_sum += neg_2p2 / (ip * ip - p2_sq);
    }

    return local_sum;
}

// Orchestrates the multi-threaded calculation.
double calculate(long long iterations, int param1, int param2) {
    // Determine the number of threads to use, based on hardware concurrency.
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<double>> futures;

    // Divide the total iterations into chunks for each thread.
    const long long chunk_size = iterations / num_threads;
    
    for (unsigned int i = 0; i < num_threads; ++i) {
        const long long start = i * chunk_size + 1;
        // The last thread takes care of any remaining iterations.
        const long long end = (i == num_threads - 1) ? iterations : (i + 1) * chunk_size;
        
        if (start > iterations) {
            continue; // Avoid launching threads for no work.
        }
        
        // Launch a new asynchronous task for each chunk. std::async manages the thread pool.
        futures.push_back(std::async(std::launch::async, calculate_chunk, start, end, param1, param2));
    }

    // Collect the results from all threads.
    double total_sum = 0.0;
    for (auto& fut : futures) {
        total_sum += fut.get(); // .get() blocks until the result is available.
    }

    // The initial value in the Python code was 1.0, to which the summation is added.
    return 1.0 + total_sum;
}

int main() {
    // Use fast I/O.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    constexpr long long iterations = 200'000'000LL;
    constexpr int param1 = 4;
    constexpr int param2 = 1;

    const auto start_time = std::chrono::high_resolution_clock::now();
    
    const double result = calculate(iterations, param1, param2) * 4.0;

    const auto end_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end_time - start_time;

    // Print results in the specified format.
    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
