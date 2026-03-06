
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>

int main() {
    auto start_time = std::chrono::steady_clock::now();

    const long long iterations = 200000000LL;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 16;  // Fallback

    long long chunk_size = iterations / num_threads;
    long long remainder = iterations % num_threads;

    std::vector<double> partial_sums(num_threads);
    std::vector<std::thread> threads;

    long long current_start = 1;
    for (unsigned int t = 0; t < num_threads; ++t) {
        long long current_end = current_start + chunk_size - 1;
        if (static_cast<long long>(t) < remainder) ++current_end;
        if (current_start > current_end) break;

        threads.emplace_back([t, current_start, current_end, &partial_sums]() {
            double local_sum = 0.0;
            for (long long i = current_start; i <= current_end; ++i) {
                double term1 = 4.0 * i - 1.0;
                double term2 = 4.0 * i + 1.0;
                double denom = term1 * term2;
                local_sum += -2.0 / denom;
            }
            partial_sums[t] = local_sum;
        });

        current_start = current_end + 1;
    }

    for (auto& th : threads) {
        th.join();
    }

    double result = 1.0;
    for (double ps : partial_sums) {
        result += ps;
    }
    result *= 4.0;

    auto end_time = std::chrono::steady_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << "\n";
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << execution_time << " seconds\n";

    return 0;
}
