#include "frame_fitter.h"

#include <cassert>
#include <limits>
#include <list>

#include "color_distances_table.h"
#include "color_table.h"
#include "image.h"
#include "opcode.h"
#include "state.h"

namespace vcsmc {

FrameFitter::FrameFitter() {
}

FrameFitter::~FrameFitter() {
}

// Number of scan lines before the start of pixel output to start issuing byte
// code for.
const uint32 kVideoReserveScanLines = 3;
const uint32 kTotalActionCount =
    (kFrameHeightPixels + kVideoReserveScanLines) * kScanLineWidthCycles;

struct Action {
  ~Action() {}
  float error;
  std::unique_ptr<State> exit_state;
  std::unique_ptr<op::OpCode> opcode;
  const Action* prior_action;
};

float FrameFitter::Fit(const uint8* half_colus) {
  printf("building frame prefix\n");
  AppendFramePrefixOpCodes();

  // Row index is the starting clock within the frame. Each row contains an
  // unordered list of all potential actions that could be taken at that time
  // step.
  std::vector<std::vector<std::unique_ptr<Action>>> actions;
  actions.reserve(kTotalActionCount);

  // If an Action at index i in |actions| finishes at time i+3, a pointer to
  // that Action will appear in the list at index i+3 in |finishing|.
  std::vector<std::vector<const Action*>> finishing;
  finishing.reserve(kTotalActionCount);

  printf("building lists\n");
  for (uint32 i = 0; i < kTotalActionCount + 3; ++i) {
    actions.push_back(std::vector<std::unique_ptr<Action>>());
    finishing.push_back(std::vector<const Action*>());
  }

  printf("building first action\n");
  // Append list of initial actions based on |starting_state|. Start with
  // changes to registers.
  for (uint32 i = 0; i < 256; ++i) {
    std::unique_ptr<Action> action = BuildAction(nullptr, makeLDA((uint8)i),
        half_colus);
    finishing[2].push_back(action.get());
    actions[0].push_back(std::move(action));
    action = BuildAction(nullptr, makeLDX((uint8)i), half_colus);
    finishing[2].push_back(action.get());
    actions[0].push_back(std::move(action));
    action = BuildAction(nullptr, makeLDY((uint8)i), half_colus);
    finishing[2].push_back(action.get());
    actions[0].push_back(std::move(action));
  }

  // Append list of stores.
  for (uint32 i = 0; i < TIA::TIA_COUNT; ++i) {
    if (!ShouldSimulateTIA((TIA)i))
      continue;
    std::unique_ptr<Action> action = BuildAction(nullptr, makeSTA((TIA)i),
        half_colus);
    finishing[3].push_back(action.get());
    actions[0].push_back(std::move(action));
    // No need to explore timings of other stores on strobes.
    if (!(kTIAStrobeMask & (1ULL << i))) {
      action = BuildAction(nullptr, makeSTX((TIA)i), half_colus);
      finishing[3].push_back(action.get());
      actions[0].push_back(std::move(action));
      action = BuildAction(nullptr, makeSTY((TIA)i), half_colus);
      finishing[3].push_back(action.get());
      actions[0].push_back(std::move(action));
    }
  }

  // Append NOP
  std::unique_ptr<Action> nop = BuildAction(nullptr, makeNOP(), half_colus);
  finishing[2].push_back(nop.get());
  actions[0].push_back(std::move(nop));

  uint64 total_actions = actions[0].size();

  for (uint32 i = 1; i < kTotalActionCount; ++i) {
    printf("starting action %d\n", i);
    // If no actions are finishing at this time we can simply skip this entire
    // row of Actions.
    if (finishing[i].size() == 0) {
      printf("skipping action %d, empty finishing array\n", i);
      continue;
    }

    // The loads are essentially NOPs in terms of state machine operation so
    // they will all naturally pick the same minimum-error entry state from the
    // finishing table. Therefore we pick it only once in advance.
    const Action* min_error_nop_action = finishing[i][0];
    for (uint32 j = 1; j < finishing[i].size(); ++j) {
      if (finishing[i][j]->error < min_error_nop_action->error)
        min_error_nop_action = finishing[i][j];
    }
    printf("minimum error prior action error is %f\n",
        min_error_nop_action->error);

    // Append NOP.
    nop = BuildAction(min_error_nop_action, makeNOP(), nullptr);
    finishing[i + 2].push_back(nop.get());
    actions[i].push_back(std::move(nop));

    // Append loads to registers.
    printf("appending loads on action %d\n", i);
    for (uint32 j = 0; j < 256; ++j) {
      std::unique_ptr<Action> action = BuildAction(
          min_error_nop_action, makeLDA((uint8)j), nullptr);
      finishing[i + 2].push_back(action.get());
      actions[i].push_back(std::move(action));
      action = BuildAction(min_error_nop_action, makeLDX((uint8)j), nullptr);
      finishing[i + 2].push_back(action.get());
      actions[i].push_back(std::move(action));
      action = BuildAction(min_error_nop_action, makeLDY((uint8)j), nullptr);
      finishing[i + 2].push_back(action.get());
      actions[i].push_back(std::move(action));
    }

    // Append stores.
    printf("appending stores on action %d\n", i);
    for (uint32 j = 0; j < TIA::TIA_COUNT; ++j) {
      if (!ShouldSimulateTIA((TIA)j))
        continue;
      std::unique_ptr<Action> sta = BuildAction(
          finishing[i][0], makeSTA((TIA)j), half_colus);
      for (uint32 k = 1; k < finishing[i].size(); ++k) {
        std::unique_ptr<Action> action = BuildAction(finishing[i][k],
            makeSTA((TIA)j), half_colus);
        if (action->error < sta->error)
          sta.swap(action);
      }
      finishing[i + 3].push_back(sta.get());
      actions[i].push_back(std::move(sta));

      if (kTIAStrobeMask & (1ULL << j))
        continue;

      std::unique_ptr<Action> stx = BuildAction(
          finishing[i][0], makeSTX((TIA)j), half_colus);
      for (uint32 k = 1; k < finishing[i].size(); ++k) {
        std::unique_ptr<Action> action = BuildAction(finishing[i][k],
            makeSTX((TIA)j), half_colus);
        if (action->error < stx->error)
          stx.swap(action);
      }
      finishing[i + 3].push_back(stx.get());
      actions[i].push_back(std::move(stx));

      std::unique_ptr<Action> sty = BuildAction(
          finishing[i][0], makeSTY((TIA)j), half_colus);
      for (uint32 k = 1; k < finishing[i].size(); ++k) {
        std::unique_ptr<Action> action = BuildAction(finishing[i][k],
            makeSTY((TIA)j), half_colus);
        if (action->error < sty->error)
          sty.swap(action);
      }
      finishing[i + 3].push_back(sty.get());
      actions[i].push_back(std::move(sty));
    }

    total_actions += actions[i].size();
    printf("finished processing action %d of %d, "
           "added %d actions to %llu total, %llu bytes.\n",
           i, kTotalActionCount, actions[i].size(), total_actions,
           total_actions * (sizeof(Action) + sizeof(State)));
  }

  assert(finishing[kTotalActionCount].size() > 0);

  // Find minimum-error action finishing at the end of the frame, and traverse
  // back to start of frame adding opcodes and states in backwards order.
  const Action* end_action = finishing[kTotalActionCount][0];
  for (uint32 i = 1; i < finishing[kTotalActionCount].size(); ++i) {
    if (finishing[kTotalActionCount][i]->error < end_action->error) {
      end_action = finishing[kTotalActionCount][i];
    }
  }

  error_ = end_action->error;
  sim_ = SimulateBack(end_action);

  std::vector<std::unique_ptr<op::OpCode>> frame_opcodes;
  while (end_action != nullptr) {
    frame_opcodes.push_back(end_action->opcode->Clone());
    end_action = end_action->prior_action;
  }

  while (frame_opcodes.size()) {
    opcodes_.push_back(frame_opcodes.back()->Clone());
    frame_opcodes.pop_back();
  }

  AppendFrameSuffixOpCodes();
  return error_;
}

std::unique_ptr<Image> FrameFitter::SimulateToImage() {
  std::unique_ptr<Image> image(
      new Image(kFrameWidthPixels, kFrameHeightPixels));
  uint32* pixel = image->pixels_writeable();
  for (uint32 i = 0; i < kFrameWidthPixels * kFrameHeightPixels; ++i) {
    *pixel = kAtariNTSCABGRColorTable[sim_[i]];
    ++pixel;
  }
  return std::move(image);
}

void FrameFitter::AppendFramePrefixOpCodes() {
  assert(opcodes_.size() == 0);

  // Turn off VBLANK, turn on VSYNC for three scanlines.
  opcodes_.push_back(makeLDA(0));
  opcodes_.push_back(makeSTA(TIA::VBLANK));
  opcodes_.push_back(makeLDA(2));
  opcodes_.push_back(makeSTA(TIA::VSYNC));
  opcodes_.push_back(makeSTA(TIA::WSYNC));
  opcodes_.push_back(makeSTA(TIA::WSYNC));
  opcodes_.push_back(makeSTA(TIA::WSYNC));

  // Turn off VSYNC, then 37 scanlines of vertical blank, occupy two scanlines
  // with initialization code, then leave some scanlines before scanout for the
  // optimization code.
  opcodes_.push_back(makeLDA(0));
  opcodes_.push_back(makeSTA(TIA::VSYNC));
  for (uint32 i = 0; i < kVBlankScanLines - (2 + kVideoReserveScanLines); ++i)
    opcodes_.push_back(makeSTA(TIA::WSYNC));

  opcodes_.push_back(makeLDA(0));            // 0
  opcodes_.push_back(makeSTA(TIA::RESP0));   // 2
  opcodes_.push_back(makeSTA(TIA::RESP1));   // 5
  opcodes_.push_back(makeSTA(TIA::NUSIZ0));  // 8
  opcodes_.push_back(makeSTA(TIA::NUSIZ1));  // 11
  opcodes_.push_back(makeSTA(TIA::COLUP0));  // 14
  opcodes_.push_back(makeSTA(TIA::COLUP1));  // 17
  opcodes_.push_back(makeSTA(TIA::COLUPF));  // 20
  opcodes_.push_back(makeSTA(TIA::COLUBK));  // 23
  opcodes_.push_back(makeSTA(TIA::CTRLPF));  // 26
  opcodes_.push_back(makeSTA(TIA::REFP0));   // 29
  opcodes_.push_back(makeSTA(TIA::REFP1));   // 32
  opcodes_.push_back(makeSTA(TIA::PF0));     // 35
  opcodes_.push_back(makeSTA(TIA::PF1));     // 38
  opcodes_.push_back(makeSTA(TIA::PF2));     // 41
  opcodes_.push_back(makeSTA(TIA::AUDC0));   // 44
  opcodes_.push_back(makeSTA(TIA::AUDC1));   // 47
  opcodes_.push_back(makeSTA(TIA::AUDF0));   // 50
  opcodes_.push_back(makeSTA(TIA::AUDF1));   // 53
  opcodes_.push_back(makeSTA(TIA::AUDV0));   // 56
  opcodes_.push_back(makeSTA(TIA::AUDV1));   // 59
  opcodes_.push_back(makeSTA(TIA::GRP0));    // 62
  opcodes_.push_back(makeSTA(TIA::GRP1));    // 65
  opcodes_.push_back(makeSTA(TIA::ENAM0));   // 68
  opcodes_.push_back(makeSTA(TIA::ENAM1));   // 71
  opcodes_.push_back(makeNOP());             // 74

  opcodes_.push_back(makeSTA(TIA::ENABL));   // 0
  opcodes_.push_back(makeSTA(TIA::VDELP0));  // 3
  opcodes_.push_back(makeSTA(TIA::VDELP1));  // 6
  opcodes_.push_back(makeSTA(TIA::VDELBL));  // 9
  opcodes_.push_back(makeSTA(TIA::RESMP0));  // 12
  opcodes_.push_back(makeSTA(TIA::RESMP1));  // 15
  opcodes_.push_back(makeSTA(TIA::HMCLR));   // 18
  opcodes_.push_back(makeLDX(0));            // 20
  opcodes_.push_back(makeLDY(0));            // 22
  opcodes_.push_back(makeSTA(TIA::WSYNC));   // 24

  // Now compute states, to give the optimization code something to start
  // working from.
  states_.reserve(opcodes_.size() + 2);
  states_.push_back(std::unique_ptr<State>(new State()));
  for (uint32 i = 0; i < opcodes_.size(); ++i) {
    states_.push_back(opcodes_[i]->Transform(states_.rbegin()->get()));
  }
}

void FrameFitter::AppendFrameSuffixOpCodes() {
  // Turn on VBLANK, then 30 scanlines of overscan.
  opcodes_.push_back(makeLDA(2));
  opcodes_.push_back(makeSTA(TIA::VBLANK));
  for (uint32 i = 0; i < kOverscanScanLines; ++i) {
    opcodes_.push_back(makeSTA(TIA::WSYNC));
  }
}

std::unique_ptr<Action> FrameFitter::BuildAction(
    const Action* prior_action,
    std::unique_ptr<op::OpCode> opcode,
    const uint8* half_colus) {
  std::unique_ptr<Action> action(new Action());
  action->opcode = std::move(opcode);

  if (prior_action) {
    action->exit_state = action->opcode->Transform(
        prior_action->exit_state.get());
  } else {
    State* starting_state = states_.rbegin()->get();
    action->exit_state = action->opcode->Transform(starting_state);
  }
  action->prior_action = prior_action;
  if (half_colus) {
    std::unique_ptr<uint8[]> sim = SimulateBack(action.get());

    float accum = 0.0f;
    for (uint32 i = 0; i < kFrameWidthPixels * kFrameHeightPixels; ++i) {
      accum +=
          kAtariColorDistances[(half_colus[i] * kNTSCColors) + sim[i]];
    }
    action->error = accum;
  } else {
    action->error = prior_action->error;
  }
  return std::move(action);
}

std::unique_ptr<uint8[]> FrameFitter::SimulateBack(const Action* last_action) {
  std::list<std::unique_ptr<State>> back_states;
  std::unique_ptr<uint8[]> sim(
      new uint8[kFrameWidthPixels * kFrameHeightPixels]);
  const Action* back_action = last_action;
  while (back_action) {
    back_states.push_front(back_action->exit_state->Clone());
    back_action = back_action->prior_action;
  }
  for (std::list<std::unique_ptr<State>>::iterator it = back_states.begin();
      it != back_states.end(); ++it) {
    (*it)->ColorInto(sim.get());
  }
  return std::move(sim);
}

bool FrameFitter::ShouldSimulateTIA(TIA tia) {
  switch(tia) {
//    case COLUP0:
//    case COLUP1:
//    case COLUPF:
    case COLUBK:
//    case CTRLPF:
//    case PF0:
//    case PF1:
//    case PF2:
//    case RESP0:
//    case RESP1:
//    case GRP0:
//    case GRP1:
      return true;

    default:
      return false;
  }
}

}  // namespace vcsmc
