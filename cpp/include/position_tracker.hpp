#pragma once
#include <vector>
#include <string>
#include <mlx/mlx.h>

namespace tsp {

class SparsePositionTracker {
public:
    SparsePositionTracker();
    void step(int num_tokens = 1);
    void prune(const std::vector<int>& island_ids, int compressed_index = -1);
    mlx::core::array get_positions() const;
    const std::vector<int>& get_position_ids() const { return position_ids_; }

private:
    std::vector<int> position_ids_;
    int current_pos_;
};

} // namespace tsp
