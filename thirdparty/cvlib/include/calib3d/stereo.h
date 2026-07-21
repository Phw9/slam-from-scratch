// Rectified stereo geometry: closed-form triangulation with gating and
// stereo rectification for calibrated horizontal rigs.

#ifndef CVLIB_CALIB3D_STEREO_H_
#define CVLIB_CALIB3D_STEREO_H_

#include "../types.h"
#include "../error_codes.h"
#include "../ransac_params.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

// Shared RANSAC parameters; see cvlib/ransac_params.h.
using RansacParams = ::cvlib::RansacParams;

/*
Triangulates matched pixels from a rectified stereo pair in closed form.

Convention (shared with the bundle-adjustment stereo rows): the right
camera sits at +baseline along the left camera x-axis, so
u_right = u_left - fx * baseline / z. Per correspondence, the epipolar
gate |v_left - v_right| <= max_epipolar_error and the disparity gate
min_disparity <= (u_left - u_right) <= max_disparity must pass; passing
rows get depth z = fx * baseline / disparity back-projected into the
left camera frame, failing rows are zeroed and masked out.

@param left_pixels Left-view pixels, N-by-2.
@param right_pixels Right-view pixels, N-by-2.
@param k Shared intrinsics (3x3).
@param baseline Stereo baseline (> 0), same unit as the output points.
@param max_epipolar_error Row-difference gate (>= 0), pixels.
@param min_disparity Lower disparity gate (> 0), pixels.
@param max_disparity Upper disparity gate (>= min_disparity), pixels.
@param points_cam Output left-camera-frame points, N-by-3.
@param inlier_mask Output length-N array (1 = triangulated, 0 = gated).
@param num_inliers Output count of triangulated points.
@returns ErrorCode.
*/
ErrorCode triangulate_rectified_stereo(
    const Matrix* left_pixels, const Matrix* right_pixels,
    const Matrix* k, float64_t baseline,
    float64_t max_epipolar_error, float64_t min_disparity,
    float64_t max_disparity,
    Matrix* points_cam, int32_t* inlier_mask, int32_t* num_inliers);

/*
Computes rectifying rotations and projections for a calibrated stereo
rig with a predominantly horizontal baseline.

The relative rotation is split evenly between the two cameras (each is
rotated halfway), then both are rotated so the baseline lies along the
new x-axis; after applying r1/r2 the two image planes are coplanar and
matched points share a row. Distortion inputs are accepted for
interface completeness but do not affect the rotations; distortion is
consumed by the caller's undistort/remap step. The new shared
intrinsics use the mean fy as focal length and the mean principal
point of the two cameras.

Rig convention: r and t map left-camera coordinates to right-camera
coordinates (x_right = r * x_left + t). Rigs whose rectified baseline
is more vertical than horizontal are rejected with kInvalidArgument.

@param k1 Left intrinsics (3x3).
@param d1 Optional left distortion coefficients; may be null.
@param k2 Right intrinsics (3x3).
@param d2 Optional right distortion coefficients; may be null.
@param r Relative rotation (3x3, orthonormal).
@param t Relative translation, length 3 (nonzero).
@param r1 Output rectifying rotation for the left camera (3x3).
@param r2 Output rectifying rotation for the right camera (3x3).
@param p1 Output left projection in the rectified frame (3x4).
@param p2 Output right projection in the rectified frame (3x4).
@param q Output disparity-to-depth reprojection matrix (4x4):
       (u, v, disparity, 1) maps to homogeneous left-rectified 3D.
@returns ErrorCode.
*/
ErrorCode stereo_rectify(const Matrix* k1, const Vector* d1,
                         const Matrix* k2, const Vector* d2,
                         const Matrix* r, const Vector* t,
                         Matrix* r1, Matrix* r2, Matrix* p1, Matrix* p2,
                         Matrix* q);

/*
Reprojects a dense disparity map to 3D through the 4x4 Q matrix from
stereo_rectify: each pixel maps as (u, v, d, 1) -> Q * (u, v, d, 1)
and dehomogenizes into the left-rectified camera frame. Pixels with
non-positive, non-finite, or sentinel (-1) disparities, or a
degenerate homogeneous scale, are zeroed and masked out.

@param disparity Disparity map, rows-by-cols (invalid entries <= 0).
@param q Reprojection matrix (4x4, finite).
@param points_cam Output points, (rows*cols)-by-3 in row-major pixel
       order (row index = v * cols + u).
@param valid_mask Output length rows*cols (1 = reprojected, 0 = invalid).
@param num_valid Output count of reprojected pixels.
@returns ErrorCode.
*/
ErrorCode reproject_disparity(const Matrix* disparity, const Matrix* q,
                              Matrix* points_cam, int32_t* valid_mask,
                              int32_t* num_valid);

/*
Fits the ground plane from camera-frame 3D points with robust plane
fitting and an orientation gate.

Rows selected by valid_mask (nonzero = use; null = all rows) are fed to
the RANSAC plane fit, so the zeroed rows of triangulate_rectified_stereo
or reproject_disparity chain in directly with their masks. The fitted
plane a*x + b*y + c*z + d = 0 (||[a,b,c]|| = 1) is oriented so the
normal points along expected_normal; when the angle between them
exceeds max_tilt_angle the consensus surface is not the ground and
kNotConverged is returned.

@param points_cam Camera-frame points, N-by-3.
@param valid_mask Optional length-N selection (nonzero = use); may be null.
@param expected_normal Expected plane normal in the camera frame,
       length 3 (nonzero, e.g. (0, -1, 0) for a forward camera).
@param max_tilt_angle Maximum normal deviation (radians, in [0, pi]).
@param params RANSAC parameters; inlier_thresh is the point-plane distance.
@param plane_out Output plane [a, b, c, d], length 4 (refit on inliers,
       oriented along expected_normal).
@param inlier_mask Output 0/1 mask over the N input rows (masked-out rows
       are 0); may be null.
@param num_inliers Output number of inliers used for the final fit.
@returns ErrorCode (kNotConverged when no consensus reaches min_inliers
         or the consensus plane fails the tilt gate; outputs are then
         not meaningful).
*/
ErrorCode fit_ground_plane(const Matrix* points_cam,
                           const int32_t* valid_mask,
                           const Vector* expected_normal,
                           float64_t max_tilt_angle,
                           RansacParams params,
                           Vector* plane_out, int32_t* inlier_mask,
                           int32_t* num_inliers);

/*
Computes the homography mapping image pixels to a metric bird's-eye-view
grid over the ground plane (inverse perspective mapping).

The ground frame is built from the camera-frame plane a*x + b*y + c*z
+ d = 0: origin at the foot of the perpendicular from the camera
center, forward axis (ground Y) along the optical axis projected onto
the plane, lateral axis (ground X) completing a right-handed frame
with the plane normal. The BEV grid samples ground X in [x_min, x_max]
and ground Y in [y_min, y_max] at resolution length-units per pixel,
with forward pointing up: u_bev = (X - x_min) / resolution and
v_bev = (y_max - Y) / resolution. The output maps homogeneous image
pixels to BEV pixels and feeds image::warp_perspective directly.

Cameras on the plane or looking along the plane normal (no forward
direction on the ground) are rejected with kInvalidArgument.

@param k Camera intrinsics (3x3).
@param ground_plane Plane [a, b, c, d] in the camera frame, length 4
       (either orientation; same length unit as the grid extents).
@param x_min Lateral grid minimum.
@param x_max Lateral grid maximum (> x_min).
@param y_min Forward grid minimum.
@param y_max Forward grid maximum (> y_min).
@param resolution Ground length per BEV pixel (> 0).
@param h_out Output image-to-BEV homography (3x3).
@returns ErrorCode.
*/
ErrorCode bev_homography(const Matrix* k, const Vector* ground_plane,
                         float64_t x_min, float64_t x_max,
                         float64_t y_min, float64_t y_max,
                         float64_t resolution, Matrix* h_out);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_STEREO_H_
