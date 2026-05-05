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
    std::vector<double> y(N);
    
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

    benchmark("softmax_naive", softmax_naive, x, y, warmup, repeat);
    benchmark("softmax_online", softmax_online_branch, x, y, warmup, repeat);

    return 0;
}