#include "mssim.h"

#include <memory>

#include "constants.h"

namespace vcsmc {

// These three constants should total to 1.0.
const double kLabLWeight = 0.8;
const double kLabaWeight = 0.1;
const double kLabbWeight = 0.1;

// How many pixels to consider in a square around each pixel.
const uint32 kWindowSize = 8;

// Constants defined in the Wang et al paper to keep the SSIM value stable
// around small values (where a divide by zero might be otherwise possible).
const double kC1 = (0.01 * 100.0) * (0.01 * 100.0);
const double kC2 = (0.03 * 100.0) * (0.03 * 100.0);

// TODO: optimize.
double Mssim(const double* lab_a, const double* lab_b, uint32 image_width,
    uint32 image_height) {
  // First we compute the local means of each pixel channel in both images. The
  // local mean averages the pixel at x,y in each channel with the
  // kWindowSize - 1 pixels to the right and below. We elide computation of
  // pixels with less than a full window.
  if (image_width < kWindowSize || image_height < kWindowSize)
    return 0.0;

  uint32 mssim_image_width = image_width - kWindowSize + 1;
  uint32 mssim_image_height = image_height - kWindowSize + 1;
  uint32 mssim_values = mssim_image_width * mssim_image_height;

  std::unique_ptr<double> means_a(new double[mssim_values * 3]);
  std::unique_ptr<double> means_b(new double[mssim_values * 3]);

  double* mu_a = means_a.get();
  double* mu_b = means_b.get();
  for (uint32 i = 0; i < mssim_image_height; ++i) {
    for (uint32 j = 0; j < mssim_image_width; ++j) {
      const double* a = lab_a + (((i * image_width) + j) * 4);
      const double* b = lab_b + (((i * image_width) + j) * 4);
      *(mu_a)     = 0.0;
      *(mu_a + 1) = 0.0;
      *(mu_a + 2) = 0.0;
      *(mu_b)     = 0.0;
      *(mu_b + 1) = 0.0;
      *(mu_b + 2) = 0.0;
      for (uint32 k = 0; k < kWindowSize; ++k) {
        for (uint32 l = 0; l < kWindowSize; ++l) {
          *(mu_a) += *a;
          ++a;
          *(mu_a + 1) += *a;
          ++a;
          *(mu_a + 2) += *a;
          a += 2;
          *(mu_b) += *b;
          ++b;
          *(mu_b + 1) += *b;
          ++b;
          *(mu_b + 2) += *b;
          b += 2;
        }
        a += (image_width - kWindowSize) * 4;
        b += (image_width - kWindowSize) * 4;
      }
      *(mu_a) /= static_cast<double>(kWindowSize * kWindowSize);
      *(mu_a + 1) /= static_cast<double>(kWindowSize * kWindowSize);
      *(mu_a + 2) /= static_cast<double>(kWindowSize * kWindowSize);
      *(mu_b) /= static_cast<double>(kWindowSize * kWindowSize);
      *(mu_b + 1) /= static_cast<double>(kWindowSize * kWindowSize);
      *(mu_b + 2) /= static_cast<double>(kWindowSize * kWindowSize);
      mu_a += 3;
      mu_b += 3;
    }
  }

  // Now that we have the means we use that to compute the squared standard
  // deviation and the covariance in each window, following the same procedure
  // as above.
  std::unique_ptr<double> stddev_a(new double[mssim_values * 3]);
  std::unique_ptr<double> stddev_b(new double[mssim_values * 3]);
  std::unique_ptr<double> covariance_ab(new double[mssim_values * 3]);

  double* std_a = stddev_a.get();
  double* std_b = stddev_b.get();
  double* cov_ab = covariance_ab.get();
  for (uint32 i = 0; i < mssim_image_height; ++i) {
    for (uint32 j = 0; j < mssim_image_width; ++j) {
      const double* a = lab_a + (((i * image_width) + j) * 4);
      const double* b = lab_b + (((i * image_width) + j) * 4);
      mu_a = means_a.get() + (((i * mssim_image_width) + j) * 3);
      mu_b = means_b.get() + (((i * mssim_image_width) + j) * 3);
      *(std_a)     = 0.0;
      *(std_a + 1) = 0.0;
      *(std_a + 2) = 0.0;
      *(std_b)     = 0.0;
      *(std_b + 1) = 0.0;
      *(std_b + 2) = 0.0;
      for (uint32 k = 0; k < kWindowSize; ++k) {
        for (uint32 l = 0; l < kWindowSize; ++l) {
          double a_del_L = *a - *mu_a;
          ++a;
          ++mu_a;
          double a_del_a = *a - *mu_a;
          ++a;
          ++mu_a;
          double a_del_b = *a - *mu_a;
          a += 2;
          ++mu_a;
          double b_del_L = *b - *mu_b;
          ++b;
          ++mu_b;
          double b_del_a = *b - *mu_b;
          ++b;
          ++mu_b;
          double b_del_b = *b - *mu_b;
          b += 2;
          ++mu_b;
          *(std_a) += (a_del_L * a_del_L);
          *(std_a + 1) += (a_del_a * a_del_a);
          *(std_a + 2) += (a_del_b * a_del_b);
          *(std_b) += (b_del_L * b_del_L);
          *(std_b + 1) += (b_del_a * b_del_a);
          *(std_b + 2) += (b_del_b * b_del_b);
          *(cov_ab) = (a_del_L * b_del_L);
          *(cov_ab + 1) = (a_del_a * b_del_a);
          *(cov_ab + 2) = (a_del_b * b_del_b);
        }
        a += (image_width - kWindowSize) * 4;
        b += (image_width - kWindowSize) * 4;
        mu_a += (mssim_image_width - kWindowSize) * 3;
        mu_b += (mssim_image_width - kWindowSize) * 3;
      }
      *(std_a) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(std_a + 1) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(std_a + 2) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(std_b) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(std_b + 1) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(std_b + 2) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(cov_ab) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(cov_ab + 1) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      *(cov_ab + 2) /= static_cast<double>((kWindowSize * kWindowSize) - 1);
      std_a += 3;
      std_b += 3;
      cov_ab += 3;
    }
  }

  // We now have all the pieces to compute the SSI at each pixel and return the
  // mean value.
  double L_mssim = 0.0;
  double a_mssim = 0.0;
  double b_mssim = 0.0;
  mu_a = means_a.get();
  mu_b = means_b.get();
  std_a = stddev_a.get();
  std_b = stddev_b.get();
  cov_ab = covariance_ab.get();
  for (uint32 i = 0; i < mssim_image_width * mssim_image_height; ++i) {
    L_mssim += (((2 * *mu_a * *mu_b) + kC1) * ((2 * *cov_ab) + kC2)) /
        (((*mu_a * *mu_a) + (*mu_b * *mu_b) + kC1) *
            ((*std_a * *std_a) + (*std_b * *std_b) + kC2));
    ++mu_a;
    ++mu_b;
    ++std_a;
    ++std_b;
    ++cov_ab;
    a_mssim += (((2 * *mu_a * *mu_b) + kC1) * ((2 * *cov_ab) + kC2)) /
        (((*mu_a * *mu_a) + (*mu_b * *mu_b) + kC1) *
            ((*std_a * *std_a) + (*std_b * *std_b) + kC2));
    ++mu_a;
    ++mu_b;
    ++std_a;
    ++std_b;
    ++cov_ab;
    b_mssim += (((2 * *mu_a * *mu_b) + kC1) * ((2 * *cov_ab) + kC2)) /
        (((*mu_a * *mu_a) + (*mu_b * *mu_b) + kC1) *
            ((*std_a * *std_a) + (*std_b * *std_b) + kC2));
    ++mu_a;
    ++mu_b;
    ++std_a;
    ++std_b;
    ++cov_ab;
  }
  L_mssim /= static_cast<double>(mssim_image_width * mssim_image_height);
  a_mssim /= static_cast<double>(mssim_image_width * mssim_image_height);
  b_mssim /= static_cast<double>(mssim_image_width * mssim_image_height);

  return (L_mssim * kLabLWeight) +
         (a_mssim * kLabaWeight) +
         (b_mssim * kLabbWeight);
}

}  // namespace vcsmc
