#include "player_fitter.h"

#include <cassert>

#include "adjacency_map.h"
#include "bit_map.h"
#include "spec.h"

namespace vcsmc {

PlayerFitter::PlayerFitter() {
}

PlayerFitter::~PlayerFitter() {
}

void PlayerFitter::FindOptimumPath(const BitMap* bitmap, bool favor_right) {
  // It is assumed we are dealing with a bitmap of same dimensions as the VCS
  // graphics screen.
  assert(kFrameWidthPixels == bitmap->width());
  assert(kFrameHeightPixels == bitmap->height());

  // Since we are interested in maximizing the number of pixels covered by the
  // 8 pixel wide player graphics object we build a map of the number of pixels
  // in the saliency map |bitmap| for each pixel + 7 to the right of that pixel.
  AdjacencyMap adj_map;
  adj_map.Build(bitmap);

  std::unique_ptr<uint32[]> coverage_map(
      new uint32[kFrameWidthPixels * kFrameHeightPixels]);
  std::unique_ptr<uint32[]> progression_map(
      new uint32[kFrameWidthPixels * kFrameHeightPixels]);
  std::memset(coverage_map.get(), 0, kFrameSizeBytes * sizeof(uint32));
  std::memset(progression_map.get(), 0, kFrameSizeBytes * sizeof(uint32));

  // Coverage scores for the bottom row are initialized to be the value from
  // the adjacency map on that row.
  uint32* coverage_row = coverage_map.get() +
      ((kFrameHeightPixels - 1) * kFrameWidthPixels);
  for (uint32 i = 6; i < 160; i += 3)
    coverage_row[i] = adj_map.count_at(i, kFrameHeightPixels - 1);

  // Now we traverse the coverage map from the bottom up less one (as we've
  // already computed the coverage for the bottom row above), computing the best
  // increase in coverage for the next line up and storing it in the coverage
  // map.
  for (uint32 i = 1; i < kFrameHeightPixels; ++i) {
    uint32 y = kFrameHeightPixels - 1 - i;
    coverage_row = coverage_map.get() + (y * kFrameWidthPixels);
    uint32* next_coverage_row = coverage_row + kFrameWidthPixels;
    uint32* progression_row = progression_map.get() + (y * kFrameWidthPixels);

    // For each position we can reach on the current row, we calculate the total
    // coverage that would result in scheduling the player for that position,
    // and then scheduling the player for each position on the next line. We
    // record the max coverage that would result.
    for (uint32 j = 6; j < 160; j += 3) {
      uint32 max_coverage = 0;
      uint32 j_coverage = adj_map.count_at(j, y);
      for (uint32 k = 0; k < 160; k += 3) {
        uint32 k_coverage = next_coverage_row[k];
        // Issuing a reset on this line before the position at j will cause the
        // VCS to skip rendering any player graphics for the j row. So we only
        // add coverage for this row if the reset for the position of the next
        // row will occur after this row is rendered.
        if (k >= j)
          k_coverage += j_coverage;

        if ((k_coverage > max_coverage) ||
            (favor_right && (k_coverage == max_coverage))) {
          max_coverage = k_coverage;
          progression_row[j] = k;
        }
      }
      coverage_row[j] = max_coverage;
    }
  }

  // Now we extract the optimum path from top to bottop and store the masks and
  // offsets per-row.
  row_masks_.reset(new uint8[kFrameHeightPixels]);
  row_offsets_.reset(new uint32[kFrameHeightPixels]);
  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    uint32* coverage_row = coverage_map.get() + (i * kFrameWidthPixels);
    uint32* progression_row = progression_map.get() + (i * kFrameWidthPixels);
    uint32 max_coverage = *coverage_row;
    uint32 max_offset = *progression_row;
    for (uint32 j = 1; j < kFrameWidthPixels - 7; ++j) {
      if (coverage_row[j] > max_coverage ||
          (favor_right && (coverage_row[j] == max_coverage))) {
        max_coverage = coverage_row[j];
        max_offset = progression_row[j];
      }
    }

    // Save max offset.
    row_offsets_[i] = max_offset;

    uint8 bitmask = 0;
    if (bitmap->bit(max_offset + 0, i)) bitmask |= 0x01;
    if (bitmap->bit(max_offset + 1, i)) bitmask |= 0x02;
    if (bitmap->bit(max_offset + 2, i)) bitmask |= 0x04;
    if (bitmap->bit(max_offset + 3, i)) bitmask |= 0x08;
    if (bitmap->bit(max_offset + 4, i)) bitmask |= 0x10;
    if (bitmap->bit(max_offset + 5, i)) bitmask |= 0x20;
    if (bitmap->bit(max_offset + 6, i)) bitmask |= 0x40;
    if (bitmap->bit(max_offset + 7, i)) bitmask |= 0x80;

    row_masks_[i] = bitmask;
  }
}

std::unique_ptr<BitMap> PlayerFitter::MakeCoverageMap() const {
  assert(row_offsets_);
  assert(row_masks_);

  std::unique_ptr<BitMap> bm(new BitMap(kFrameWidthPixels, kFrameHeightPixels));
  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    uint32 offset = row_offsets_[i];
    for (uint32 j = 0; j < kFrameWidthPixels; ++j) {
      if (j >= offset && j < offset + 8) {
        bm->SetBit(j, i, row_masks_[i] & (1 << (j - offset)));
      } else {
        bm->SetBit(j, i, false);
      }
    }
  }

  return std::move(bm);
}

// This lovely table is copied from Andrew Towers, who wrote some very detailed
// notes about the TIA. I found a copy here:
// http://www.atarihq.com/danb/files/TIA_HW_Notes.txt
//
// CPU  CLK Pixel  Main Close Medium  Far  PF
//
// 0      0  -  1    17    33    65    -
// ...
// 22    66  -  1    17    33    65    -
// 22.6 --------------------------------------------------------
// 23    69     1     6    22    38    70  0.25
// 24    72     4     9    25    41    73  1
// 25    75     7    12    28    44    76  1.75
// 26    78    10    15    31    47    79  2.5
// 27    81    13    18    34    50    82  3.25
// 28    84    16    21    37    53    85  3
// 29    87    19    24    40    56    88
// 30    90    22    27    43    59    91
// 31    93    25    30    46    62    94
// 32    96    28    33    49    65    97
// 33    99    31    36    52    68   100
// 34   102    34    39    55    71   103
// 35   105    37    42    58    74   106
// 36   108    40    45    61    77   109
// 37   111    43    48    64    80   112
// 38   114    46    51    67    83   115
// 39   117    49    54    70    86   118
// 40   120    52    57    73    89   121
// 41   123    55    60    76    92   124
// 42   126    58    63    79    95   127
// 43   129    61    66    82    98   130
// 44   132    64    69    85   101   133
// 45   135    67    72    88   104   136
// 46   138    70    75    91   107   139
// 47   141    73    78    94   110   142
// 48   144    76    81    97   113   145
// 49   147    79    84   100   116   148
// 50   150    82    87   103   119   151
// 51   153    85    90   106   122   154
// 52   156    88    93   109   125   157
// 53   159    91    96   112   128     0
// 54   162    94    99   115   131     3
// 55   165    97   102   118   134     6
// 56   168   100   105   121   137     9
// 57   171   103   108   124   140    12
// 58   174   106   111   127   143    15
// 59   177   109   114   130   146    18
// 60   180   112   117   133   149    21
// 61   183   115   120   136   152    24
// 62   186   118   123   139   155    27
// 63   189   121   126   142   158    30
// 64   192   124   129   145     1    33
// 65   195   127   132   148     4    36
// 66   198   130   135   151     7    39
// 67   201   133   138   154    10    42
// 68   204   136   141   157    13    45
// 69   207   139   144     0    16    48
// 70   210   142   147     3    19    51
// 71   213   145   150     6    22    54
// 72   216   148   153     9    25    57
// 73   219   151   156    12    28    60
// 74   222   154   159    15    31    63
// 75   225   157     2    18    34    66
// 76   228     0     5    21    37    69
// ----------------------------------------------------- Start HBLANK
//

void PlayerFitter::AppendSpecs(std::vector<Spec>* specs,
      bool is_player_one) const {
  TIA respx = is_player_one ? TIA::RESP1 : TIA::RESP0;
  TIA grpx = is_player_one ? TIA::GRP1 : TIA::GRP0;
  // The ith item in |row_offsets_| indicates when the reset should be strobed
  // at the (i - 1)th scanline. Both players start at unknown positions and
  // must be reset, so we put them at a position off the screen.
  uint32 previous_position = kFrameWidthPixels;
  uint32 scanline_start =
      (kVSyncScanLines + kVBlankScanLines - 1) * kScanLineWidthClocks;
  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    uint32 current_position = row_offsets_[i];
    if (previous_position != current_position) {
      // From the player graphics timing table it's apparent that positioning
      // the player graphics at pixel x requires issuing a reset that finishes
      // on the previous line plus 63 clocks plus the start pixel offset.
      uint32 reset_clock = scanline_start + 63 + current_position;
      specs->push_back(Spec(respx, 0, Range(reset_clock, reset_clock)));
      current_position = previous_position;
    }
    scanline_start += kScanLineWidthClocks;
    uint32 graphics_start = scanline_start + kHBlankWidthClocks +
        current_position;
    specs->push_back(Spec(grpx, row_masks_[i],
        Range(graphics_start, graphics_start + 8)));
  }
}

}  // namespace vcsmc
