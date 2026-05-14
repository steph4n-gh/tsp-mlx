#pragma once
#include <mlx/mlx.h>
#include <optional>

namespace tsp {

mlx::core::array process_attention(
    mlx::core::array queries,
    mlx::core::array keys,
    std::optional<mlx::core::array> mask = std::nullopt
);

} // namespace tsp
