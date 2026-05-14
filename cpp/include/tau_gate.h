#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct {
    char** nodes;
    size_t nodes_count;
    double tau;
    double connectivity_score;
} FFIPartitionResult;

FFIPartitionResult* tau_gate_analyze(
    const int* edges_ptr,
    size_t edges_count,
    const char** nodes_ptr,
    size_t nodes_count
);

void tau_gate_free_result(FFIPartitionResult* ptr);

#ifdef __cplusplus
}
#endif
