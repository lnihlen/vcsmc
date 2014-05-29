#include "background_color_strategy.h"

#include <cassert>

#include "color.h"
#include "opcode.h"
#include "pallette.h"
#include "pixel_strip.h"
#include "range.h"
#include "schedule.h"
#include "spec.h"
#include "state.h"

namespace vcsmc {

std::unique_ptr<Schedule> BackgroundColorStrategy::Fit(
    const PixelStrip* target_strip, const Schedule* starting_schedule) {
  const Pallette* pallette = target_strip->pallette(1);

  // Choose best-fit single color.
  uint8 bg_color = pallette->colu(0);

  std::unique_ptr<Schedule> schedule(new Schedule(*starting_schedule));
  // All Specs for bg color must be met before we start drawing pixels.

  uint32 pixel_start_time =
      (kScanLineWidthClocks * target_strip->row_id()) + kHBlankWidthClocks;
  Range range(0, pixel_start_time);

  // There are a couple of ways to make the VCS paint a uniform line of color.
  // We could turn off the players by setting GRP0 and GRP1 to 0. If one or
  // both of them are currently zero this is cheap. We could set the COLUP0 or
  // COLUP1 to the background color. If one or both of these is already in the
  // bg color this is cheap. Same thing with playfield, although setting the
  // COLUPF to the bg color seems cheaper than wiping out the three PF
  // registers, unless 2 or 3 of them are already zero.

  // Note that we always try to add Specs in descending order of impact to the
  // strategy. Rule of thumb is how many pixels would be impacted by not meeting
  // the Spec. So we set BG color first, then wipe out the playfield, then wipe
  // out the player graphics, etc.
  Spec colubk(TIA::COLUBK, bg_color, range);
  schedule->AddSpec(colubk);

  std::list<Spec> zero_playfield;
  zero_playfield.push_back(Spec(TIA::PF0, 0, range));
  zero_playfield.push_back(Spec(TIA::PF1, 0, range));
  zero_playfield.push_back(Spec(TIA::PF2, 0, range));
  uint32 zpf_cost = schedule->CostToAddSpecs(&zero_playfield);

  Spec colupf(TIA::COLUPF, bg_color, range);
  uint32 colupf_cost = schedule->CostToAddSpec(colupf);
  if (zpf_cost < colupf_cost) {
    schedule->AddSpecs(&zero_playfield);
  } else {
    schedule->AddSpec(colupf);
  }

  std::list<Spec> zero_player_graphics;
  zero_player_graphics.push_back(Spec(TIA::VDELP0, 0, range));
  zero_player_graphics.push_back(Spec(TIA::VDELP1, 0, range));
  zero_player_graphics.push_back(Spec(TIA::GRP0, 0, range));
  zero_player_graphics.push_back(Spec(TIA::GRP1, 0, range));
  uint32 zpg_cost = schedule->CostToAddSpecs(&zero_player_graphics);

  std::list<Spec> set_player_colors;
  set_player_colors.push_back(Spec(TIA::COLUP0, bg_color, range));
  ste_player_colors.push_back(Spec(TIA::COLUP1, bg_color, range));
  uint32 spc_cost = schedule->CostToAddSpecs(&spc_cost);
  if (zpg_cost < spc_cost) {
    schedule->AddSpecs(&zero_player_graphics);
  } else {
    schedule->AddSpecs(&set_player_colors);
  }

  // TODO: missiles and ball.

  return schedule;
}

}  // namespace vcsmc
