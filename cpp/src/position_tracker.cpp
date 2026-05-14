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

void SparsePositionTracker::prune(const std::vector<int>& island_ids) {
    std::set<int> island_set(island_ids.begin(), island_ids.end());
    position_ids_.erase(
        std::remove_if(position_ids_.begin(), position_ids_.end(),
            [&island_set](int id) { return island_set.count(id) > 0; }),
        position_ids_.end()
    );
}

mlx::core::array SparsePositionTracker::get_positions() const {
    return mlx::core::array(position_ids_.data(), {static_cast<int>(position_ids_.size())}, mlx::core::int32);
}

} // namespace tsp
