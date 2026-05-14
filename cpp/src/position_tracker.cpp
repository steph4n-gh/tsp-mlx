#include "position_tracker.hpp"
#include <algorithm>
#include <set>

namespace tsp {

SparsePositionTracker::SparsePositionTracker() : current_pos_(0) {}

void SparsePositionTracker::step(int num_tokens) {
    for (int i = 0; i < num_tokens; ++i) {
        position_ids_.push_back(current_pos_++);
    }
}

void SparsePositionTracker::prune(const std::vector<int>& island_ids, int compressed_index) {
    std::set<int> island_set(island_ids.begin(), island_ids.end());
    std::vector<int> new_positions;
    for (int pid : position_ids_) {
        if (island_set.count(pid) == 0) {
            new_positions.push_back(pid);
        } else if (compressed_index != -1 && pid == compressed_index) {
            new_positions.push_back(pid);
        }
    }
    position_ids_ = new_positions;
}

mlx::core::array SparsePositionTracker::get_positions() const {
    return mlx::core::array(position_ids_.data(), {static_cast<int>(position_ids_.size())}, mlx::core::int32);
}

} // namespace tsp
