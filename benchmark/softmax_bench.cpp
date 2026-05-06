#include <omp.h>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>

using Clock = std::chrono::steady_clock;

void softmax_naive(const std::vector<double>& x, std::vector<double>& y) {
    double m = x[0];

    for (double e : x) {
        if (m < e) {
            m = e;
        }
    }

    double sum = 0.0;

    for (double e : x) {
        sum += std::exp(e - m);
    }

    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - m) / sum;
    }
}

void softmax_online(const std::vector<double>& x, std::vector<double>& y) {
    double m = -std::numeric_limits<double>::infinity();
    double sum = 0.0;

    for (double e : x) {
        double prev_m = m;
        m = std::max(m, e);

        sum = sum * std::exp(prev_m - m) + std::exp(e - m);
    }

    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - m) / sum;
    }
}

void softmax_online_branch(const std::vector<double>& x, std::vector<double>& y) {
    double m = x[0];
    double sum = 1.0;

    for (std::size_t i = 1; i < x.size(); ++i) {
        double v = x[i];

        if (v <= m) {
            sum += std::exp(v - m);
        } else {
            sum = sum * std::exp(m - v) + 1.0;
            m = v;
        }
    }

    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - m) / sum;
    }
}

struct MD {
    double m;
    double sum;
};

static inline MD identity_md() {
    return {-std::numeric_limits<double>::infinity(), 0.0};
}

static inline MD combine_md(const MD& a, const MD& b) {
    if (a.sum == 0.0) return b;
    if (b.sum == 0.0) return a;

    double m = std::max(a.m, b.m);
    double sum = a.sum * std::exp(a.m - m)
               + b.sum * std::exp(b.m - m);

    return {m, sum};
}

static inline MD online_chunk(const std::vector<double>& x,
                              std::size_t begin,
                              std::size_t end) {
    if (begin >= end) return identity_md();

    double m = x[begin];
    double sum = 1.0;

    for (std::size_t i = begin + 1; i < end; ++i) {
        double v = x[i];

        if (v <= m) {
            sum += std::exp(v - m);
        } else {
            sum = sum * std::exp(m - v) + 1.0;
            m = v;
        }
    }

    return {m, sum};
}

void softmax_online_parallel_omp(const std::vector<double>& x,
                                 std::vector<double>& y) {
    std::size_t n = x.size();
    y.resize(n);

    if (n == 0) return;

    MD total = identity_md();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        std::size_t begin = n * tid / nt;
        std::size_t end   = n * (tid + 1) / nt;

        MD local = online_chunk(x, begin, end);

        #pragma omp critical
        {
            total = combine_md(total, local);
        }
    }

    double m = total.m;
    double sum = total.sum;

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        y[i] = std::exp(x[i] - m) / sum;
    }
}

template <typename Func>
void benchmark(const std::string& name, Func func, const std::vector<double>& x, std::vector<double>& y, int warmup, int repeat) {
    volatile double sink = 0.0;

    // warm-up
    for (int i = 0; i < warmup; ++i) {
        func(x, y);
        sink += y[i % y.size()];
    }

    auto start = Clock::now();

    for (int i = 0; i < repeat; ++i) {
        func(x, y);
        sink += y[i % y.size()];
    }

    auto end = Clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    double total_ms = elapsed.count();
    double avg_ms = total_ms / repeat;
    double ns_per_elem = avg_ms * 1e6 / x.size();

    std::cout << name << "\n";
    std::cout << "  total time   : " << total_ms << " ms\n";
    std::cout << "  avg time     : " << avg_ms << " ms\n";
    std::cout << "  ns / element : " << ns_per_elem << " ns\n";
    std::cout << "  sink         : " << sink << "\n\n";
}


int main() {
    const int N = 1 << 20;
    const int warmup = 10;
    const int repeat = 100;

    std::vector<double> x(N);
    std::vector<double> y1(N);
    std::vector<double> y2(N);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0,10.0);
    for (double& e : x) e = dist(rng);

    std::vector<double> y_naive(N);
    std::vector<double> y_online(N);
    softmax_naive(x, y_naive);
    softmax_online_branch(x, y_online);

    double sum_naive = 0.0;
    double sum_online = 0.0;
    double max_abs_diff = 0.0;

    for (int i = 0; i < N; ++i) {
    sum_naive += y_naive[i];
    sum_online += y_online[i];
    max_abs_diff = std::max(max_abs_diff, std::abs(y_naive[i] - y_online[i]));
    }
    std::cout << "sum_naive      = " << sum_naive << "\n";
    std::cout << "sum_online     = " << sum_online << "\n";
    std::cout << "max_abs_diff   = " << max_abs_diff << "\n";

    benchmark("softmax_naive", softmax_naive, x, y1, warmup, repeat);
    benchmark("softmax_online_branch", softmax_online_branch, x, y2, warmup, repeat);
    benchmark("softmax_online_parallel", softmax_online_parallel_omp, x, y2, warmup, repeat);

    return 0;
}