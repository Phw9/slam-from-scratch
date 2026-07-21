// Brute-force descriptor matching for feature2d descriptors.

#ifndef CVLIB_FEATURE2D_MATCHING_H_
#define CVLIB_FEATURE2D_MATCHING_H_

#include "../error_codes.h"
#include "../defs.h"

#include <cstdint>
#include <vector>

namespace cvlib {
namespace feature2d {

/*
A single descriptor correspondence.

@param query_idx Row index into the query descriptor set.
@param train_idx Row index into the train descriptor set.
@param distance Match distance (Hamming bits or L2).
*/
struct DescriptorMatch {
    int32_t query_idx = -1;
    int32_t train_idx = -1;
    float64_t distance = 0.0;
};

/*
Filters applied after nearest-neighbour search.

@param ratio Lowe ratio test in (0, 1]; 0 disables. A match is kept when
  best < ratio * second_best. With fewer than two train descriptors the
  test cannot be evaluated and the match is rejected.
@param cross_check Keep only mutual nearest neighbours when true.
@param max_distance Keep matches with distance <= max_distance when > 0;
  0 disables the cutoff.
*/
struct DescriptorMatchOptions {
    float64_t ratio = 0.0;
    bool cross_check = false;
    float64_t max_distance = 0.0;
};

/*
Returns pass-through matching defaults (plain nearest neighbour).

@returns DescriptorMatchOptions.
*/
DescriptorMatchOptions default_descriptor_match_options();

/*
Brute-force Hamming matching for binary descriptors (e.g. ORB, N-by-32
bytes row-major). For each query row the nearest train row wins;
ties resolve deterministically to the lowest train index.

@param query Query descriptors, num_query * descriptor_bytes bytes.
@param num_query Number of query descriptors.
@param train Train descriptors, num_train * descriptor_bytes bytes.
@param num_train Number of train descriptors.
@param descriptor_bytes Bytes per descriptor (> 0).
@param options Optional filters; null uses defaults.
@param matches Output matches ordered by ascending query index.
@returns ErrorCode.
*/
ErrorCode match_hamming_brute_force(const uint8_t* query, int32_t num_query,
                                    const uint8_t* train, int32_t num_train,
                                    int32_t descriptor_bytes,
                                    const DescriptorMatchOptions* options,
                                    std::vector<DescriptorMatch>* matches);

/*
Brute-force L2 matching for float descriptors (e.g. SIFT, N-by-128
row-major). For each query row the nearest train row wins; ties resolve
deterministically to the lowest train index.

@param query Query descriptors, num_query * descriptor_size floats.
@param num_query Number of query descriptors.
@param train Train descriptors, num_train * descriptor_size floats.
@param num_train Number of train descriptors.
@param descriptor_size Elements per descriptor (> 0).
@param options Optional filters; null uses defaults.
@param matches Output matches ordered by ascending query index.
@returns ErrorCode.
*/
ErrorCode match_l2_brute_force(const float32_t* query, int32_t num_query,
                               const float32_t* train, int32_t num_train,
                               int32_t descriptor_size,
                               const DescriptorMatchOptions* options,
                               std::vector<DescriptorMatch>* matches);

}  // namespace feature2d
}  // namespace cvlib

#endif  // CVLIB_FEATURE2D_MATCHING_H_
