// Image core: non-owning views and owning images with rows, cols,
// stride, interleaved channels, pixel format, and ROI sub-views.

#ifndef CVLIB_IMAGE_IMAGE_H_
#define CVLIB_IMAGE_IMAGE_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace image {

// Pixel formats. Samples are stored row-major with interleaved
// channels; stride counts SAMPLES per row (stride >= cols * channels).
static constexpr int32_t kFormatF64 = 0;
static constexpr int32_t kFormatF32 = 1;
static constexpr int32_t kFormatU8  = 2;

static constexpr int32_t kMaxImageChannels = 4;

/*
Non-owning, read-only image view.

@param data First sample of the first pixel (format-typed storage).
@param rows Row count (> 0).
@param cols Column count (> 0).
@param stride Samples per row (>= cols * channels).
@param channels Interleaved channel count (1..4).
@param format kFormatF64 | kFormatF32 | kFormatU8.
*/
struct ImageView {
    const void* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
    int32_t channels = 0;
    int32_t format = kFormatF64;
};

/*
Owning image; create with image_create and release with image_destroy.
Storage is dense (stride == cols * channels).
*/
struct Image {
    void* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
    int32_t channels = 0;
    int32_t format = kFormatF64;
};

/*
Returns the size of one sample of the format in bytes (0 for unknown
formats).
*/
int32_t image_sample_bytes(int32_t format);

/*
Allocates a zero-filled dense image.

@param rows Row count (> 0).
@param cols Column count (> 0).
@param channels Channel count (1..4).
@param format Pixel format.
@param out Output image; overwritten on success.
@returns ErrorCode.
*/
ErrorCode image_create(int32_t rows, int32_t cols, int32_t channels,
                       int32_t format, Image* out);

/*
Releases image storage and resets the struct; safe on empty images.
*/
void image_destroy(Image* img);

/*
Borrows a read-only view of an owning image.
*/
ImageView image_view(const Image* img);

/*
Borrows a stride-preserving rectangular sub-view (ROI). The view
aliases the source storage; it stays valid only while the source does.

@param src Source view.
@param row0 Top row of the ROI (>= 0).
@param col0 Left column of the ROI (>= 0).
@param rows ROI row count (> 0, row0 + rows <= src->rows).
@param cols ROI column count (> 0, col0 + cols <= src->cols).
@param out Output view.
@returns ErrorCode.
*/
ErrorCode image_roi(const ImageView* src, int32_t row0, int32_t col0,
                    int32_t rows, int32_t cols, ImageView* out);

/*
Copies src into dst, converting the sample format when they differ.
dst must be pre-created with the same rows, cols, and channels.
u8 <-> float conversions keep numeric values (no 1/255 scaling);
float-to-u8 clamps to [0, 255] and rounds to nearest.

@param src Source view.
@param dst Destination image (same shape, any supported format).
@returns ErrorCode.
*/
ErrorCode image_convert(const ImageView* src, Image* dst);

/*
Reads one sample as float64 regardless of the stored format.

@param view Source view.
@param row Row index in [0, rows).
@param col Column index in [0, cols).
@param channel Channel index in [0, channels).
@param value_out Output sample value.
@returns ErrorCode.
*/
ErrorCode image_at(const ImageView* view, int32_t row, int32_t col,
                   int32_t channel, float64_t* value_out);

/*
Writes one sample from float64 into an owning image (clamped and
rounded for u8 storage).

@param img Destination image.
@param row Row index in [0, rows).
@param col Column index in [0, cols).
@param channel Channel index in [0, channels).
@param value Sample value.
@returns ErrorCode.
*/
ErrorCode image_set(Image* img, int32_t row, int32_t col, int32_t channel,
                    float64_t value);

}  // namespace image
}  // namespace cvlib

#endif  // CVLIB_IMAGE_IMAGE_H_
