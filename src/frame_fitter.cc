#include "frame_fitter.h"

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "color_distances_table.h"
#include "color_table.h"
#include "image.h"
#include "line_kernel.h"
#include "opcode.h"
#include "random.h"
#include "state.h"

namespace vcsmc {

// When evaluating a Line program, how many scanlines should the program be
// scored against? Should be at least 1, to evaluate the line the program is
// running on, but can be more.
const uint32 kLinesToScore = 2;
const uint32 kGenerationSize = 50;
const uint32 kBoutSize = kGenerationSize / 10;
const uint32 kMaxGenerations = 5;
const float kMaxError = 100.0f;

// F000 - F400: Preamble code, up to start of frame.
// F400 - F800: Bank 0.
// F800 - FD00: Bank 1.
// FD00 - FFF7: Epilogue code, really just a placeholder for the data below.
// FFF7 - FFFC: The string "VCSMC"
// FFFC - FFFD: CPU entry point (16bit pointer)
// FFFE - FFFF: CPU Breakpoint (16bit pointer)
// const uint16 kPreambleAddress = 0xf000;
const uint16 kBank0Address = 0xf400;
const uint16 kBank1Address = 0xf800;
// const uint16 kEpilogueAddress = 0xfd00;
const uint32 kBankSize = 1024;


FrameFitter::FrameFitter() {
}

FrameFitter::~FrameFitter() {
}

float FrameFitter::Fit(const uint8* half_colus) {
  AppendFramePrefixOpCodes();
  uint32 seed[] = {
      0xcc02f8c5, 0xa0997c34, 0xa86303df, 0xac1d3f0d,
      0x2508f747, 0xec381a4f, 0x65e47935, 0xddbc2287,
      0x29e1672b, 0xf6a6bed3, 0x27b13937, 0xbce6348c
  };
  Random random(seed);

  std::unique_ptr<State> entry_state = (*states_.rbegin())->Clone();
  std::vector<std::unique_ptr<LineKernel>> lines;
  float error = 0.0f;

  // Use Evolutionary Programming to fit the individual scanlines, adding the
  // best final fit LineKernel to |lines|.
  for (uint32 i = 0; i < kFrameHeightPixels + 1; ++i) {
    printf("fitting line %d of %d..\n", i, kFrameHeightPixels + 1);
    uint32 scan_line = kVSyncScanLines + kVBlankScanLines + i - 1;
    assert(scan_line * kScanLineWidthClocks ==
        entry_state->range().start_time());
    std::unique_ptr<LineKernel> lk =
        FitLine(&random, half_colus, scan_line, entry_state.get());
    entry_state = lk->exit_state()->Clone();
    uint32 next_scan_line_start_time = (scan_line + 1) * kScanLineWidthClocks;
    if (entry_state->range().start_time() < next_scan_line_start_time) {
      entry_state = entry_state->AdvanceTime(next_scan_line_start_time -
          entry_state->range().start_time());
    }
    error += lk->sim_error();
    lines.push_back(std::move(lk));
  }

  // Now for bank-fitting. We must break the line kernels into 1 KB chunks. The
  // ends of each line program have gone unterminated, to allow us room to
  // add JMP instructions where needed, to fire our bank switching algorithm.
  // We start by summing up the size of the preamble opcodes already appended
  // to the last bank.
  std::vector<std::unique_ptr<op::OpCode>>* opcodes = (*banks_.rbegin()).get();
  uint32 bank_size = 0;
  for (uint32 i = 0; i < opcodes->size(); ++i)
    bank_size += opcodes->at(i)->bytes();

  // For now we can assume that every line is 73 cycles or shorter, leaving us
  // room to append a JMP instruction on any line. Could envision a dynamic
  // programming algorithm later to pack lines into banks, if some of them are
  // too long to fit a JMP in to.
  bool bank0 = false;
  for (uint32 i = 0; i < lines.size(); ++i) {
    assert(lines[i]->total_cycles() == 73 || lines[i]->total_cycles() <= 71);
    // Append current line opcodes to bank.
    bank_size += lines[i]->total_bytes();
    assert(bank_size < kBankSize);
    lines[i]->Append(opcodes, &states_);
    uint32 line_cycles = lines[i]->total_cycles();
    // We add 6 cycles to allow for enough bytes to terminate both this line
    // and the next.
    if (i == lines.size() - 1 ||
        bank_size + 6 + lines[i + 1]->total_bytes() > kBankSize) {
      // Append a JMP and swap banks.
      std::unique_ptr<op::OpCode> jmp = makeJMP(
          bank0 ? kBank0Address : kBank1Address);
      bank0 = !bank0;
      assert(bank_size + jmp->bytes() < kBankSize);
      states_.push_back(jmp->Transform((*states_.rbegin()).get()));
      line_cycles += jmp->cycles();
      opcodes->push_back(std::move(jmp));
      banks_.push_back(
          std::unique_ptr<std::vector<std::unique_ptr<op::OpCode>>>(
              new std::vector<std::unique_ptr<op::OpCode>>()));
      opcodes = (*banks_.rbegin()).get();
      bank_size = 0;
    }

    // Do we still need to terminate the line?
    if (line_cycles < 76) {
      assert(line_cycles != 75);
      std::unique_ptr<op::OpCode> term;
      if (lines[i]->total_cycles() == 74) {
        term = makeNOP();
      } else {
        term = makeSTA(TIA::WSYNC);
      }
      states_.push_back(term->Transform((*states_.rbegin()).get()));
      bank_size += term->bytes();
      opcodes->push_back(std::move(term));
    }
  }

  AppendFrameSuffixOpCodes(bank0);
  return error;
}

std::unique_ptr<Image> FrameFitter::SimulateToImage() {
  std::unique_ptr<Image> image(
      new Image(kFrameWidthPixels, kFrameHeightPixels));
  for (uint32 i = 0; i < states_.size(); ++i)
    states_[i]->PaintInto(image.get());
  return std::move(image);
}

void FrameFitter::SaveBinary(const char* file_name) {
  int bin_fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (bin_fd < 0)
    return;
  std::unique_ptr<uint8[]> bin_buf(new uint8[kBankSize]);
  for (uint32 i = 0; i < banks_.size(); ++i) {
    std::memset(bin_buf.get(), 0, kBankSize);
    uint8* buf_ptr = bin_buf.get();
    for (uint32 j = 0; j < banks_[i]->size(); ++j)
      buf_ptr += banks_[i]->at(j)->bytecode(buf_ptr);
    write(bin_fd, bin_buf.get(), kBankSize);
  }

  close(bin_fd);
}

void FrameFitter::AppendFramePrefixOpCodes() {
  assert(banks_.size() == 0);
  std::unique_ptr<std::vector<std::unique_ptr<op::OpCode>>> opcodes(
      new std::vector<std::unique_ptr<op::OpCode>>());

  // Turn off VBLANK, turn on VSYNC for three scanlines.
  opcodes->push_back(makeLDA(0));
  opcodes->push_back(makeSTA(TIA::VBLANK));
  opcodes->push_back(makeLDA(2));
  opcodes->push_back(makeSTA(TIA::VSYNC));
  opcodes->push_back(makeSTA(TIA::WSYNC));
  opcodes->push_back(makeSTA(TIA::WSYNC));
  opcodes->push_back(makeSTA(TIA::WSYNC));

  // Turn off VSYNC, then 37 scanlines of vertical blank, occupy two scanlines
  // with initialization code.
  opcodes->push_back(makeLDA(0));
  opcodes->push_back(makeSTA(TIA::VSYNC));
  for (uint32 i = 0; i < kVBlankScanLines - 3; ++i)
    opcodes->push_back(makeSTA(TIA::WSYNC));

  opcodes->push_back(makeLDA(0));              // 0
  opcodes->push_back(makeSTA(TIA::RESP0));     // 2
  opcodes->push_back(makeSTA(TIA::RESP1));     // 5
  opcodes->push_back(makeSTA(TIA::NUSIZ0));    // 8
  opcodes->push_back(makeSTA(TIA::NUSIZ1));    // 11
  opcodes->push_back(makeSTA(TIA::COLUP0));    // 14
  opcodes->push_back(makeSTA(TIA::COLUP1));    // 17
  opcodes->push_back(makeSTA(TIA::COLUPF));    // 20
  opcodes->push_back(makeSTA(TIA::COLUBK));    // 23
  opcodes->push_back(makeSTA(TIA::CTRLPF));    // 26
  opcodes->push_back(makeSTA(TIA::REFP0));     // 29
  opcodes->push_back(makeSTA(TIA::REFP1));     // 32
  opcodes->push_back(makeSTA(TIA::PF0));       // 35
  opcodes->push_back(makeSTA(TIA::PF1));       // 38
  opcodes->push_back(makeSTA(TIA::PF2));       // 41
  opcodes->push_back(makeSTA(TIA::AUDC0));     // 44
  opcodes->push_back(makeSTA(TIA::AUDC1));     // 47
  opcodes->push_back(makeSTA(TIA::AUDF0));     // 50
  opcodes->push_back(makeSTA(TIA::AUDF1));     // 53
  opcodes->push_back(makeSTA(TIA::AUDV0));     // 56
  opcodes->push_back(makeSTA(TIA::AUDV1));     // 59
  opcodes->push_back(makeSTA(TIA::GRP0));      // 62
  opcodes->push_back(makeSTA(TIA::GRP1));      // 65
  opcodes->push_back(makeSTA(TIA::ENAM0));     // 68
  opcodes->push_back(makeSTA(TIA::ENAM1));     // 71
  opcodes->push_back(makeNOP());               // 74

  opcodes->push_back(makeSTA(TIA::ENABL));     // 0
  opcodes->push_back(makeSTA(TIA::VDELP0));    // 3
  opcodes->push_back(makeSTA(TIA::VDELP1));    // 6
  opcodes->push_back(makeSTA(TIA::VDELBL));    // 9
  opcodes->push_back(makeSTA(TIA::RESMP0));    // 12
  opcodes->push_back(makeSTA(TIA::RESMP1));    // 15
  opcodes->push_back(makeSTA(TIA::HMCLR));     // 18
  opcodes->push_back(makeLDX(0));              // 20
  opcodes->push_back(makeLDY(0));              // 22
  opcodes->push_back(makeSTA(TIA::WSYNC));     // 27

  banks_.push_back(std::move(opcodes));

  // Now compute states, to give the optimization code something to start
  // working from.
  states_.push_back(std::unique_ptr<State>(new State()));
  for (uint32 i = 0; i < banks_.size(); ++i) {
    for (uint32 j = 0; j < banks_[i]->size(); ++j) {
      states_.push_back(banks_[i]->at(j)->Transform(states_.rbegin()->get()));
    }
  }
}

void FrameFitter::AppendFrameSuffixOpCodes(bool bank0) {
  std::vector<std::unique_ptr<op::OpCode>>* opcodes = (*banks_.rbegin()).get();
  // Turn on VBLANK, then 30 scanlines of overscan.
  opcodes->push_back(makeLDA(2));
  opcodes->push_back(makeSTA(TIA::VBLANK));
  for (uint32 i = 0; i < kOverscanScanLines; ++i) {
    opcodes->push_back(makeSTA(TIA::WSYNC));
  }

  opcodes->push_back(makeJMP(bank0 ? kBank0Address : kBank1Address));

  // Could break epilogue into multiple banks if needed.
  uint32 total_bank_size = 0;
  for (uint32 i = 0; i < opcodes->size(); ++i)
    total_bank_size += opcodes->at(i)->bytes();
  assert(total_bank_size < kBankSize);
}

std::unique_ptr<LineKernel> FrameFitter::FitLine(Random* random,
                                                 const uint8* half_colus,
                                                 uint32 scan_line,
                                                 const State* entry_state) {
  std::vector<std::unique_ptr<LineKernel>> population;
  population.reserve(2 * kGenerationSize);

  // Generate initial generation of programs.
  for (uint32 i = 0; i < kGenerationSize; ++i) {
    std::unique_ptr<LineKernel> lk(new LineKernel());
    lk->Randomize(random);
    lk->Simulate(half_colus, scan_line, entry_state, kLinesToScore);
    population.push_back(std::move(lk));
  }

  uint32 best = 0;
  uint32 generation_count = 0;
  while (generation_count < kMaxGenerations &&
      population[best]->sim_error() > kMaxError) {
    // Each member of current population generates one offspring by cloning
    // followed by mutation.
    for (uint32 i = 0; i < kGenerationSize; ++i) {
      std::unique_ptr<LineKernel> lk = population[i]->Clone();
      lk->Mutate(random);
      population.push_back(std::move(lk));
    }

    for (uint32 i = 0; i < kGenerationSize; ++i)
      population[i]->ResetVictories();

    // Simulate all the new offspring.
    for (uint32 i = kGenerationSize; i < population.size(); ++i) {
      population[i]->Simulate(
          half_colus, scan_line, entry_state, kLinesToScore);
    }

    // Compete to survive!
    for (uint32 i = 0; i < population.size(); ++i) {
      for (uint32 j = 0; j < kBoutSize; ++j) {
        population[i]->Compete(
            population[random->Next() % population.size()].get());
      }
    }

    // Sort by victories greatest to least, then remove the lower half of the
    // population.
    std::sort(population.begin(),
              population.end(),
              &FrameFitter::CompareKernels);
    population.erase(population.begin() + kGenerationSize, population.end());

    best = 0;
    uint32 worst = 0;
    float sum = population[0]->sim_error();
    for (uint32 i = 1; i < population.size(); ++i) {
      sum += population[i]->sim_error();
      if (population[i]->sim_error() < population[best]->sim_error())
        best = i;
      if (population[i]->sim_error() > population[worst]->sim_error())
        worst = i;
    }

    float mean = sum / (float)(population.size());

    printf("  gen %d, best: %f, best cycles: %d, mean: %f, worst: %f\n",
        generation_count,
        population[best]->sim_error(),
        population[best]->total_cycles(),
        mean,
        population[worst]->sim_error());
    ++generation_count;
  }

  std::unique_ptr<LineKernel> best_fit;
  // Save the best final fit from the vector, simulation results included.
  best_fit.swap(population[best]);
  return std::move(best_fit);
}

// static
bool FrameFitter::CompareKernels(const std::unique_ptr<LineKernel>& lk1,
                                 const std::unique_ptr<LineKernel>& lk2) {
  return lk1->victories() > lk2->victories();
}

}  // namespace vcsmc
