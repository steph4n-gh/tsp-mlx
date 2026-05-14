#include "sparse_cache.hpp"
#include "tau_gate.h"
#include <iostream>
#include <set>
#include <string>

namespace tsp {

KVCacheManager::KVCacheManager(double threshold) : threshold_(threshold) {}

KVCacheManager::Decision KVCacheManager::evaluate_attention(
    const mlx::core::array& attention_matrix,
    const std::vector<int>& sinks
) {
    // 1. Collapse heads (mean across axis 1)
    auto a_2d = mlx::core::mean(attention_matrix, std::vector<int>{1}); // [B, Lq, Lk]
    
    // Looking at batch 0
    int Lq = a_2d.shape(1);
    int Lk = a_2d.shape(2);
    a_2d = mlx::core::slice(a_2d, {0, 0, 0}, {1, Lq, Lk});
    a_2d = mlx::core::squeeze(a_2d, 0); // [Lq, Lk]
    
    // Handle [1, S] case for decoding
    auto a_sq = (a_2d.shape(0) == 1) ? mlx::core::squeeze(a_2d, 0) : a_2d;

    const auto& position_ids = tracker_.get_position_ids();

    if (a_sq.ndim() == 2) {
        // PREFILL
        auto a_sym = mlx::core::maximum(a_sq, mlx::core::transpose(a_sq));
        auto thresholded = mlx::core::greater(a_sym, mlx::core::array(threshold_));
        mlx::core::eval(thresholded);
        
        // Extract edges (CPU side)
        auto data = thresholded.data<bool>();
        int rows = thresholded.shape(0);
        int cols = thresholded.shape(1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (data[i * cols + j] && i != j && i < (int)position_ids.size() && j < (int)position_ids.size()) {
                    edges_.insert({position_ids[i], position_ids[j]});
                }
            }
        }
    } else {
        // DECODE
        auto thresholded = mlx::core::greater(a_sq, mlx::core::array(threshold_));
        mlx::core::eval(thresholded);
        
        auto data = thresholded.data<bool>();
        int source_abs = position_ids.back();
        for (int i = 0; i < thresholded.size(); ++i) {
            if (data[i]) {
                if (i < (int)position_ids.size()) {
                    int target_abs = position_ids[i];
                    if (source_abs != target_abs) {
                        edges_.insert({source_abs, target_abs});
                        edges_.insert({target_abs, source_abs});
                    }
                }
            }
        }
    }

    // Call Rust static library
    std::vector<int> edges_flat;
    for (const auto& edge : edges_) {
        edges_flat.push_back(edge.first);
        edges_flat.push_back(edge.second);
    }

    std::vector<std::string> node_names;
    node_names.reserve(position_ids.size());
    std::vector<const char*> node_ptrs;
    node_ptrs.reserve(position_ids.size());
    for (int id : position_ids) {
        node_names.push_back(std::to_string(id));
        node_ptrs.push_back(node_names.back().c_str());
    }

    auto result = tau_gate_analyze(
        edges_flat.data(),
        edges_.size(),
        node_ptrs.data(),
        node_ptrs.size()
    );

    Decision decision;
    decision.action = "ALLOW";
    if (result) {
        // Threshold for structural isolation
        if (result->connectivity_score < 0.1) { 
             decision.action = "GARBAGE_COLLECT";
             std::set<int> sink_set(sinks.begin(), sinks.end());
             for (size_t i = 0; i < result->nodes_count; ++i) {
                 if (result->nodes[i] == nullptr) continue;
                 try {
                     int node_id = std::stoi(result->nodes[i]);
                     // Defensive Hard-Stop: Never prune a sink
                     if (sink_set.count(node_id) == 0) {
                         decision.island_indices.push_back(node_id);
                     }
                 } catch (...) {
                     // Ignore malformed node IDs
                 }
             }
        }
        tau_gate_free_result(result);
    }

    return decision;
}

std::vector<std::pair<mlx::core::array, mlx::core::array>> KVCacheManager::update(
    const mlx::core::array& attention_matrix,
    const std::vector<std::pair<mlx::core::array, mlx::core::array>>& kv_caches,
    const std::vector<int>& sinks
) {
    auto decision = evaluate_attention(attention_matrix, sinks);
    
    if (decision.action == "GARBAGE_COLLECT" && !decision.island_indices.empty()) {
        std::cout << "[TSP] 🧹 GARBAGE COLLECT: Evicting " << decision.island_indices.size() << " tokens from KV Cache\n";
        
        std::set<int> island_set(decision.island_indices.begin(), decision.island_indices.end());
        const auto& position_ids = tracker_.get_position_ids();
        std::vector<int> keep_indices;
        for (int i = 0; i < (int)position_ids.size(); ++i) {
            if (island_set.count(position_ids[i]) == 0) {
                keep_indices.push_back(i);
            }
        }

        auto keep_array = mlx::core::array(keep_indices.data(), {static_cast<int>(keep_indices.size())}, mlx::core::int32);
        
        std::vector<std::pair<mlx::core::array, mlx::core::array>> pruned_caches;
        for (const auto& cache : kv_caches) {
            auto pk = mlx::core::take(cache.first, keep_array, 2);
            auto pv = mlx::core::take(cache.second, keep_array, 2);
            pruned_caches.push_back({pk, pv});
        }
        
        tracker_.prune(decision.island_indices);
        
        // Prune edges
        for (auto it = edges_.begin(); it != edges_.end(); ) {
            if (island_set.count(it->first) || island_set.count(it->second)) {
                it = edges_.erase(it);
            } else {
                ++it;
            }
        }
        
        return pruned_caches;
    }

    return kv_caches;
}

} // namespace tsp
