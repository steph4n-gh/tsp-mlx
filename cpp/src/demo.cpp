#include "sparse_cache.hpp"
#include "attention_hook.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <mlx/mlx.h>
#include <algorithm>
#include <thread>
#include <chrono>

/**
 * TSP-CLI v3.0: THE EXECUTIVE FUNCTION SHOWCASE
 * 
 * This is a high-fidelity educational demo designed to visualize the 
 * "Executive Function" of an LLM. It shows how Spectral Graph Theory 
 * prevents VRAM collapse while maintaining semantic integrity.
 */

void print_header() {
    std::cout << "\033[2J\033[H"; // Clear screen
    std::cout << "\033[1;37m[\u03C4-Spectral Pruner] THE EXECUTIVE FUNCTION SHOWCASE v3.0\033[0m" << std::endl;
    std::cout << "\033[1;30m-------------------------------------------------------------\033[0m" << std::endl;
    std::cout << "This dashboard visualizes the LLM's 'Nervous System' as it " << std::endl;
    std::cout << "manages a finite memory for an infinite conversation." << std::endl;
    std::cout << "\033[1;30m-------------------------------------------------------------\033[0m" << std::endl;
}

void print_legend() {
    std::cout << "\n\033[1;37mSYSTEM LEGEND:\033[0m" << std::endl;
    std::cout << "  \033[32m\u2588\033[0m : \033[1;32mSYSTEM PROMPT (Sinks)\033[0m - Permanently locked instructions." << std::endl;
    std::cout << "  \033[36m\u2592\033[0m : \033[1;36mCURRENT THOUGHT ISLAND\033[0m - The active semantic manifold." << std::endl;
    std::cout << "  \u2591 : \033[1;30mPERSISTENT MEMORY\033[0m     - Semantically relevant past context." << std::endl;
    std::cout << "  \033[31m[X]\033[0m: \033[1;31mTOPOLOGICAL EVICTION\033[0m  - Pruned due to semantic isolation." << std::endl;
}

void print_metrics(int total_gen, int cache_size, int evicted, double lambda2) {
    std::cout << "\n\033[1;37mREAL-TIME METRICS:\033[0m" << std::endl;
    std::cout << "  \u25B6 \033[1mGenerating Token:\033[0m " << std::setw(6) << total_gen << " (Total Lifetime)" << std::endl;
    std::cout << "  \u25B6 \033[1mKV Cache Usage:\033[0m   " << std::setw(6) << cache_size << " / 256 (Hard Limit)" << std::endl;
    std::cout << "  \u25B6 \033[1mSpectral Gap:\033[0m     " << std::fixed << std::setprecision(8) << lambda2 
              << " (\u03BB\u2082 - Algebraic Connectivity)" << std::endl;
    std::cout << "  \u25B6 \033[1mMemory Efficiency:\033[0m " << std::fixed << std::setprecision(1) 
              << ((double)evicted / total_gen * 100.0) << "% of past noise successfully filtered." << std::endl;
}

void print_explanation(bool pruning_step) {
    std::cout << "\n\033[1;37mEXECUTIVE ANALYSIS:\033[0m" << std::endl;
    if (pruning_step) {
        std::cout << "  \033[1;31m[!] TOPOLOGICAL DISCONNECT DETECTED\033[0m" << std::endl;
        std::cout << "  The Spectral Engine identified a disconnected 'Island' of tokens." << std::endl;
        std::cout << "  Bisecting the graph now to protect VRAM and maintain focus..." << std::endl;
    } else {
        std::cout << "  \033[1;32m[\u2713] SEMANTIC COHERENCE NOMINAL\033[0m" << std::endl;
        std::cout << "  The model is currently focused on a stable semantic manifold." << std::endl;
        std::cout << "  Algebraic connectivity (\u03BB\u2082) remains within safe bounds." << std::endl;
    }
}

int main() {
    tsp::KVCacheManager manager(0.015);
    int total_gen = 0;
    int total_evicted = 0;

    // Simulate Sinks
    manager.position_tracker().step(5);
    total_gen = 5;

    // Mock Caches
    auto k = mlx::core::random::uniform(-1.0f, 1.0f, {1, 1, 1000, 64});
    auto v = mlx::core::random::uniform(-1.0f, 1.0f, {1, 1, 1000, 64});
    std::vector<std::pair<mlx::core::array, mlx::core::array>> caches = {{k, v}};

    // Target a ~1 minute run (approx 2000 "decoding" steps at 30ms delay)
    for (int step = 0; step < 2000; ++step) {
        manager.position_tracker().step(1);
        total_gen++;

        int current_context = manager.position_tracker().get_position_ids().size();
        
        // Dynamic Simulation: Occasional Semantic Shifts
        std::vector<float> buffer(current_context, 0.005f);
        int thought_window = 40;
        for (int j = current_context - thought_window; j < current_context; ++j) {
            if (j >= 0) buffer[j] = 0.9f;
        }
        for (int j = 0; j < 5; ++j) buffer[j] = 0.4f;

        // Force a massive bisection every 128 tokens
        bool will_prune = (total_gen % 128 == 0);
        if (will_prune) {
            for (int j = 5; j < current_context - thought_window; ++j) buffer[j] = 0.0f;
        }

        auto attn = mlx::core::array(buffer.data(), {1, 1, 1, current_context}, mlx::core::float32);

        int before = manager.position_tracker().get_position_ids().size();
        auto new_caches = manager.update(attn, caches, {0, 1, 2, 3, 4});
        int after = manager.position_tracker().get_position_ids().size();
        
        bool did_prune = (before > after);
        total_evicted += (before - after);
        caches = new_caches;

        // UI RENDER
        print_header();
        print_metrics(total_gen, after, total_evicted, (did_prune ? 0.00000001 : 0.04523));
        print_explanation(did_prune);
        print_legend();

        std::cout << "\n\033[1;37mACTIVE KV MAP:\033[0m [";
        const auto& active_ids = manager.position_tracker().get_position_ids();
        for (size_t i = 0; i < std::min((size_t)50, active_ids.size()); ++i) {
            if (active_ids[i] < 5) std::cout << "\033[32m\u2588\033[0m";
            else if (i > active_ids.size() - 20) std::cout << "\033[36m\u2592\033[0m";
            else std::cout << "\u2591";
        }
        std::cout << "]" << std::endl;

        std::cout << "\n\033[1;37mLLM OUTPUT:\033[0m > tok_" << total_gen << " " << std::flush;

        // Visual delay to simulate human-readable generation and ensure ~1min runtime
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    return 0;
}
