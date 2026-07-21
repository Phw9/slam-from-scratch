// Polygon geometry APIs.

#ifndef CVLIB_GEOMETRY_POLYGON_H_
#define CVLIB_GEOMETRY_POLYGON_H_

#include "../types.h"
#include "../error_codes.h"

namespace cvlib {
namespace geometry {

/*
Computes 2D convex hull vertices in counter-clockwise order using Andrew's
monotone chain. Degenerate input contract: a single point returns that one
point (hull_size = 1); all-coincident or all-collinear input returns the
two extreme points along the sort order (lexicographic by x, then y), each
possibly repeated, since strictly-collinear boundary points are dropped
(the chain-building step pops on cross product <= 0, not < 0). Exact
duplicate points are likewise only implicitly removed by this same
same-or-negative-orientation pop, not filtered up front.

@param points Input points, N-by-2 (N >= 1).
@param hull_out Output hull points, capacity M-by-2; must satisfy M >= 1
  and M >= number of returned hull vertices. Unused rows are left untouched.
@param hull_size Output number of hull vertices written into hull_out.
@returns ErrorCode.
*/
ErrorCode convex_hull(const Matrix* points, Matrix* hull_out,
                      int32_t* hull_size);

/*
Tests if point is outside, on boundary, or inside polygon.

@param polygon Input polygon vertices, N-by-2 (N >= 3).
@param point Input query point, length 2.
@param result Output classification: -1 outside, 0 on edge, 1 inside.
@returns ErrorCode.
*/
ErrorCode point_in_polygon(const Matrix* polygon, const Vector* point,
                           int32_t* result);

/*
Computes the signed area of a simple polygon using the shoelace formula.

@param polygon Input polygon vertices, N-by-2 (N >= 3).
@param area Output signed area; positive for CCW, negative for CW order.
@returns ErrorCode.
*/
ErrorCode polygon_area(const Matrix* polygon, float64_t* area);

/*
Computes the centroid (geometric center) of a simple polygon.

@param polygon Input polygon vertices, N-by-2 (N >= 3).
@param centroid_out Output centroid coordinates, length 2.
@returns ErrorCode.
*/
ErrorCode polygon_centroid(const Matrix* polygon, Vector* centroid_out);

/*
Computes the perimeter (sum of edge lengths) of a closed polygon.

@param polygon Input polygon vertices, N-by-2 (N >= 2).
@param perimeter Output total perimeter length.
@returns ErrorCode.
*/
ErrorCode polygon_perimeter(const Matrix* polygon, float64_t* perimeter);

/*
Clips a convex subject polygon against a convex clip polygon
(Sutherland-Hodgman). Both polygons must be convex with vertices in CCW
order and positive area. The intersection of convex polygons with N and M
vertices has at most N + M vertices.

@param subject Subject polygon, N-by-2, convex CCW (N >= 3).
@param clip Clip polygon, M-by-2, convex CCW (M >= 3).
@param out Output intersection vertices in CCW order, capacity
  K-by-2 with K >= N + M. Unused rows are left untouched.
@param out_size Output vertex count; 0 when the interiors do not overlap.
@returns ErrorCode (kInvalidArgument for non-convex or non-CCW input).
*/
ErrorCode convex_polygon_intersection(const Matrix* subject,
                                      const Matrix* clip, Matrix* out,
                                      int32_t* out_size);

/*
Computes intersection-over-union of two convex polygons. Both polygons
must be convex with vertices in CCW order and positive area.

@param poly_a First polygon, N-by-2, convex CCW (N >= 3).
@param poly_b Second polygon, M-by-2, convex CCW (M >= 3).
@param iou Output IoU in [0, 1]; 0 when the interiors do not overlap.
@returns ErrorCode (kInvalidArgument for non-convex or non-CCW input).
*/
ErrorCode convex_polygon_iou(const Matrix* poly_a, const Matrix* poly_b,
                             float64_t* iou);

/*
Minimum-area enclosing rectangle of a 2D point set via rotating an
edge-aligned bounding box over the convex hull.

@param points Input points, N-by-2 (N >= 3, not all collinear).
@param corners Output rectangle corners in order, 4-by-2;
       pre-allocated.
@returns ErrorCode (kSingularMatrix for degenerate hulls).
*/
ErrorCode min_area_rect(const Matrix* points, Matrix* corners);

/*
Minimum enclosing circle of a 2D point set (exact, deterministic
incremental construction over the fixed input order).

@param points Input points, N-by-2 (N >= 1).
@param center_out Output circle center, length 2.
@param radius_out Output radius (>= 0).
@returns ErrorCode.
*/
ErrorCode min_enclosing_circle(const Matrix* points, Vector* center_out,
                               float64_t* radius_out);

}  // namespace geometry
}  // namespace cvlib

#endif  // CVLIB_GEOMETRY_POLYGON_H_
