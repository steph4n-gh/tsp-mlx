#pragma once
#include "position_tracker.hpp"
#include "compression.hpp"
#include "consolidation.hpp"
#include <mlx/mlx.h>
#include <vector>
#include <set>
#include <string>
#include <memory>

namespace tsp {

class KVCacheManager {
public:
    KVCacheManager(double threshold = 0.015, bool enable_compression = false, bool enable_consolidation = false, int head_dim = 64);
    
    struct Decision {
        std::string action;
        std::vector<int> island_indices;
    };

    std::vector<std::pair<mlx::core::array, mlx::core::array>> update(
        const mlx::core::array& attention_matrix,
        const std::vector<std::pair<mlx::core::array, mlx::core::array>>& kv_caches,
        const std::vector<int>& sinks
    );

    SparsePositionTracker& position_tracker() { return tracker_; }

private:
    Decision evaluate_attention(
        const mlx::core::array& attention_matrix,
        const std::vector<int>& sinks
    );

    SparsePositionTracker tracker_;
    double threshold_;
    std::set<std::pair<int, int>> edges_;
    
    bool enable_compression_;
    std::unique_ptr<SubManifoldAutoencoder> compressor_;

    bool enable_consolidation_;
    std::unique_ptr<MemoryConsolidator> consolidator_;
};

} // namespace tsp
