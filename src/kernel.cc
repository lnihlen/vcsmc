#include "kernel.h"

#include <array>
#include <cassert>
#include <cstring>
#include <random>
#include <sstream>

#include <farmhash.h>
extern "C" {
#include <libz26/libz26.h>
}

#include "assembler.h"
#include "color.h"
#include "color_distance_table.h"
#include "color_table.h"
#include "image.h"
#include "image_file.h"

namespace vcsmc {

const size_t kSimSkipLines = 23;

Kernel::Kernel(std::seed_seq& seed)
  : engine_(seed),
    specs_(new std::vector<Spec>()),
    bytecode_size_(0),
    total_dynamic_opcodes_(0),
    fingerprint_(0),
    score_valid_(false),
    score_(0.0),
    victories_(0) {
}

Kernel::Kernel(
    const std::string& random_state,
    SpecList specs,
    const std::vector<Range>& dynamic_areas,
    const std::vector<std::unique_ptr<uint8[]>>& packed_opcodes)
    : specs_(specs),
      total_dynamic_opcodes_(0) {
  std::istringstream sstream(random_state);
  sstream >> engine_;
  assert(dynamic_areas.size() == packed_opcodes.size());
  size_t spec_list_index = 0;
  size_t opcode_list_index = 0;
  uint32 current_cycle = 0;
  size_t total_byte_size = 0;
  while (current_cycle < kScreenSizeCycles) {
    uint32 next_spec_start_time = spec_list_index < specs_->size() ?
        specs_->at(spec_list_index).range().start_time() :
        kScreenSizeCycles;
    if (current_cycle == next_spec_start_time) {
      assert(spec_list_index < specs_->size());
      total_byte_size += specs_->at(spec_list_index).size();
      current_cycle = specs_->at(spec_list_index).range().end_time();
      ++spec_list_index;
    } else {
      uint32 starting_cycle = current_cycle;
      opcodes_.emplace_back();
      assert(opcode_list_index < dynamic_areas.size());
      assert(total_byte_size == dynamic_areas[opcode_list_index].start_time());
      uint8* current_byte = packed_opcodes[opcode_list_index].get();
      while (current_byte - packed_opcodes[opcode_list_index].get() <
          dynamic_areas[opcode_list_index].Duration()) {
        OpCode op = static_cast<OpCode>(*current_byte);
        size_t opcode_size = OpCodeBytes(op);
        uint8 arg1 = opcode_size > 1 ? *(current_byte + 1) : 0;
        uint8 arg2 = opcode_size > 2 ? *(current_byte + 2) : 0;
        opcodes_.back().push_back(PackOpCode(op, arg1, arg2));
        current_byte += opcode_size;
        total_byte_size += opcode_size;
        current_cycle += OpCodeCycles(op);
      }
      total_dynamic_opcodes_ += opcodes_.back().size();
      opcode_counts_.push_back(total_dynamic_opcodes_);
      opcode_ranges_.emplace_back(starting_cycle, current_cycle);
      ++opcode_list_index;
    }
  }
  RegenerateBytecode(total_byte_size);
}

bool Kernel::SaveImage(const std::string& file_name) const {
  if (!score_valid_ || !sim_frame_) return false;
  Image image(kTargetFrameWidthPixels, kFrameHeightPixels);
  uint32* pix = image.pixels_writeable();
  const uint8* sim = sim_frame_.get() + (kLibZ26ImageWidth * kSimSkipLines);
  for (size_t i = 0; i < kTargetFrameWidthPixels * kFrameHeightPixels; i += 2) {
    uint32 color;
    if (*sim >= 128) {
      color = 0xff00ff00;  // Bright green for undefined pixel.
    } else {
      assert(*sim < 128);
      color = kAtariNTSCABGRColorTable[*sim];
    }
    *pix = color;
    ++pix;
    *pix = color;
    ++pix;
    sim += 2;
  }
  return vcsmc::SaveImage(&image, file_name);
}

std::string Kernel::GetRandomState() const {
  std::ostringstream sstream;
  sstream << engine_;
  return sstream.str();
}

void Kernel::GenerateRandomKernelJob::Execute() {
  uint32 current_cycle = 0;
  size_t spec_list_index = 0;
  size_t total_byte_size = 0;
  // We reserve 3 cycles at the end for the jmp instruction taking us back to
  // the top.
  while (current_cycle < kScreenSizeCycles - 3) {
    uint32 next_spec_start_time = spec_list_index < specs_->size() ?
        specs_->at(spec_list_index).range().start_time() :
        kScreenSizeCycles - 3;
    size_t next_spec_size = spec_list_index < specs_->size() ?
        specs_->at(spec_list_index).size() : 0;
    if (current_cycle == next_spec_start_time) {
      assert(spec_list_index < specs_->size());
      // Copy the spec to the kernel speclist.
      kernel_->specs_->emplace_back(specs_->at(spec_list_index));
      current_cycle = specs_->at(spec_list_index).range().end_time();
      assert((total_byte_size % kBankSize) + next_spec_size <
             (kBankSize - kBankPadding));
      total_byte_size += next_spec_size;
      ++spec_list_index;
    } else {
      uint32 starting_cycle = current_cycle;
      kernel_->opcodes_.emplace_back();
      uint32 cycles_remaining = next_spec_start_time - current_cycle;
      size_t bytes_remaining =
        (total_byte_size % kBankSize) + kBankPadding < kBankSize ?
        kBankSize - ((total_byte_size % kBankSize) + kBankPadding) : 0;
      while (cycles_remaining > 0) {
        // Two possibilities for the generation of a jmp spec, used for bank
        // switching. First is that we have filled the bank within the
        // threshold. The second is that we are within a few cycles from the
        // beginning of the next spec, and that the addition of the next spec
        // would exceed the bank. We presume an average binary size of about
        // one byte per cycle.
        if ((bytes_remaining < kBankPadding ||
             (bytes_remaining < next_spec_size &&
                cycles_remaining < kBankPadding)) &&
             (cycles_remaining > 4 || cycles_remaining == 3)) {
          if (kernel_->opcodes_.back().size()) {
            kernel_->total_dynamic_opcodes_ += kernel_->opcodes_.back().size();
            kernel_->opcode_counts_.push_back(kernel_->total_dynamic_opcodes_);
            kernel_->opcode_ranges_.emplace_back(starting_cycle, current_cycle);
            kernel_->opcodes_.emplace_back();
          }
          kernel_->AppendJmpSpec(current_cycle, total_byte_size % kBankSize);
          const Spec& jmp_spec = kernel_->specs_->back();
          current_cycle += jmp_spec.range().Duration();
          starting_cycle = current_cycle;
          assert(cycles_remaining >= jmp_spec.range().Duration());
          cycles_remaining -= jmp_spec.range().Duration();
          total_byte_size += jmp_spec.size();
          assert(0 == total_byte_size % kBankSize);
          bytes_remaining = kBankSize - kBankPadding;
        } else {
          uint32 op = kernel_->GenerateRandomOpcode(cycles_remaining);
          uint32 cycles = OpCodeCycles(op);
          cycles_remaining -= cycles;
          current_cycle += cycles;
          size_t op_size = OpCodeBytes(op);
          bytes_remaining -= op_size;
          total_byte_size += op_size;
          kernel_->opcodes_.back().push_back(op);
        }
      }
      if (kernel_->opcodes_.back().size()) {
        kernel_->opcode_ranges_.emplace_back(starting_cycle, current_cycle);
        kernel_->total_dynamic_opcodes_ += kernel_->opcodes_.back().size();
        kernel_->opcode_counts_.push_back(kernel_->total_dynamic_opcodes_);
      } else {
        kernel_->opcodes_.pop_back();
      }
    }
  }
  assert(current_cycle == kScreenSizeCycles - 3);
  kernel_->AppendJmpSpec(current_cycle, total_byte_size % kBankSize);
  total_byte_size += kernel_->specs_->back().size();
  kernel_->RegenerateBytecode(total_byte_size);
}

void Kernel::ScoreKernelJob::Execute() {
  kernel_->SimulateAndScore(target_colors_);
}

void Kernel::MutateKernelJob::Execute() {
  // Copy the internal program state of the original kernel to the target.
  target_->specs_.reset(new std::vector<Spec>());
  std::copy(original_->specs_->begin(), original_->specs_->end(),
      std::back_inserter(*target_->specs_.get()));
  target_->opcodes_.clear();
  for (size_t i = 0; i < original_->opcodes_.size(); ++i) {
    target_->opcodes_.emplace_back();
    std::copy(original_->opcodes_[i].begin(), original_->opcodes_[i].end(),
        std::back_inserter(target_->opcodes_.back()));
  }
  target_->opcode_ranges_.clear();
  std::copy(original_->opcode_ranges_.begin(), original_->opcode_ranges_.end(),
      std::back_inserter(target_->opcode_ranges_));
  target_->bytecode_size_ = original_->bytecode_size_;

  target_->total_dynamic_opcodes_ = original_->total_dynamic_opcodes_;
  std::copy(original_->opcode_counts_.begin(), original_->opcode_counts_.end(),
      std::back_inserter(target_->opcode_counts_));

  // Now do the mutations to the target.
  for (size_t i = 0; i < number_of_mutations_; ++i)
    target_->Mutate();

  target_->RegenerateBytecode(target_->bytecode_size_);
}

void Kernel::ClobberSpecJob::Execute() {
  size_t target_spec_index = 0;
  for (size_t i = 0; i < specs_->size(); ++i) {
    while (target_spec_index < target_->specs_->size() &&
           target_->specs_->at(target_spec_index).range().start_time() <
              specs_->at(i).range().start_time()) {
      ++target_spec_index;
    }
    target_->specs_->at(target_spec_index) = specs_->at(i);
  }
  target_->RegenerateBytecode(target_->bytecode_size_);
}

uint32 Kernel::GenerateRandomOpcode(uint32 cycles_remaining) {
  // As no possible instruction lasts only one cycle we cannot support a
  // situation where only one cycle is remaining.
  assert(cycles_remaining > 1);
  // With two or four cycles remaining we must only generate two-cycle
  // instructions.
  if (cycles_remaining == 2 || cycles_remaining == 4) {
    return GenerateRandomLoad();
  } else if (cycles_remaining == 3) {
    return GenerateRandomStore();
  }

  std::uniform_int_distribution<int> distro(0, 1);
  int roll = distro(engine_);
  if (roll == 0) {
    return GenerateRandomLoad();
  }
  return GenerateRandomStore();
}

uint32 Kernel::GenerateRandomLoad() {
  std::uniform_int_distribution<uint8> arg_distro(0, 255);
  uint8 arg = arg_distro(engine_);
  std::uniform_int_distribution<int> op_distro(0, 1);
  switch (op_distro(engine_)) {
    case 0:
      return PackOpCode(OpCode::LDX_Immediate, arg, 0);

    case 1:
      return PackOpCode(OpCode::LDY_Immediate, arg, 0);

    default:
      assert(false);
      break;
  }
  assert(false);
  return 0;
}

uint32 Kernel::GenerateRandomStore() {
  std::array<uint8, 27> kValidTIA{{
    TIA::NUSIZ0,
    TIA::NUSIZ1,
    TIA::COLUP0,
    TIA::COLUP1,
    TIA::COLUPF,
    TIA::COLUBK,
    TIA::CTRLPF,
    TIA::REFP0,
    TIA::REFP1,
    TIA::PF0,
    TIA::PF1,
    TIA::PF2,
    TIA::RESP0,
    TIA::RESP1,
    TIA::RESM0,
    TIA::RESM1,
    TIA::RESBL,
    TIA::GRP0,
    TIA::GRP1,
    TIA::ENAM0,
    TIA::ENAM1,
    TIA::ENABL,
    TIA::VDELP0,
    TIA::VDELP1,
    TIA::VDELBL,
    TIA::RESMP0,
    TIA::RESMP1,
  }};
  std::uniform_int_distribution<size_t> tia_distro(0, kValidTIA.size() - 1);
  uint8 tia = kValidTIA[tia_distro(engine_)];
  std::uniform_int_distribution<int> op_distro(0, 1);
  switch (op_distro(engine_)) {
    case 0:
      return PackOpCode(OpCode::STX_ZeroPage, tia, 0);

    case 1:
      return PackOpCode(OpCode::STY_ZeroPage, tia, 0);

    default:
      assert(false);
      break;
  }
  assert(false);
  return 0;
}

void Kernel::AppendJmpSpec(uint32 current_cycle, size_t current_bank_size) {
  assert(current_bank_size <= (kBankSize - kBankPadding - 3));
  size_t bank_balance_size = kBankSize - current_bank_size;
  std::unique_ptr<uint8[]> bytecode(new uint8[bank_balance_size]);
  std::memset(bytecode.get() + 3, 0, bank_balance_size - 6);
  bytecode.get()[0] = JMP_Absolute;
  bytecode.get()[1] = 0x00;
  bytecode.get()[2] = 0xf0;
  // Need to create the jump table at the end of each 4K bank, pointing back to
  // the top of the program address space.
  bytecode.get()[bank_balance_size - 6] = 0x00;
  bytecode.get()[bank_balance_size - 5] = 0xf0;
  bytecode.get()[bank_balance_size - 4] = 0x00;
  bytecode.get()[bank_balance_size - 3] = 0xf0;
  bytecode.get()[bank_balance_size - 2] = 0x00;
  bytecode.get()[bank_balance_size - 1] = 0xf0;
  specs_->emplace_back(
      Range(current_cycle, current_cycle + 3),
      bank_balance_size,
      std::move(bytecode));
}

void Kernel::RegenerateBytecode(size_t bytecode_size) {
  bytecode_size_ = bytecode_size;
  score_valid_ = false;
  score_ = 0.0;
  bytecode_.reset(new uint8[bytecode_size]);
  dynamic_areas_.clear();
  uint32 current_cycle = 0;
  size_t current_range_index = 0;
  size_t current_spec_index = 0;
  uint8* current_byte = bytecode_.get();
  while (current_cycle < kScreenSizeCycles) {
    uint32 next_spec_start_time = current_spec_index < specs_->size() ?
      specs_->at(current_spec_index).range().start_time() :
      kInfinity;
    uint32 next_range_start_time = current_range_index < opcode_ranges_.size() ?
      opcode_ranges_[current_range_index].start_time() :
      kInfinity;
    if (current_cycle == next_range_start_time) {
      std::vector<uint32>& ops = opcodes_[current_range_index];
      assert(ops.size() > 0);
      uint32_t starting_byte = current_byte - bytecode_.get();
      // Serialize the packed opcodes into the buffer.
      for (size_t i = 0; i < ops.size(); ++i) {
        current_byte += UnpackOpCode(ops[i], current_byte);
        current_cycle += OpCodeCycles(ops[i]);
      }
      dynamic_areas_.emplace_back(starting_byte,
          current_byte - bytecode_.get());
      ++current_range_index;
    } else {
      assert(current_cycle == next_spec_start_time);
      // Copy the spec into the buffer.
      std::memcpy(current_byte, specs_->at(current_spec_index).bytecode(),
          specs_->at(current_spec_index).size());
      current_byte += specs_->at(current_spec_index).size();
      current_cycle = specs_->at(current_spec_index).range().end_time();
      ++current_spec_index;
    }
  }
  fingerprint_ = util::Hash64(reinterpret_cast<const char*>(bytecode_.get()),
                              bytecode_size_);
}

void Kernel::SimulateAndScore(const uint8* target_colors) {
  sim_frame_.reset(new uint8[kLibZ26ImageSizeBytes]);
  std::memset(sim_frame_.get(), 0, kLibZ26ImageSizeBytes);
  simulate_single_frame(bytecode_.get(), bytecode_size_, sim_frame_.get());
  score_ = 0.0;
  uint8* frame_pointer = sim_frame_.get() + (kLibZ26ImageWidth * kSimSkipLines);
  for (size_t i = 0; i < kFrameSizeBytes; ++i) {
    if (*frame_pointer >= 128) {
      score_ += 1000.0;
    } else {
      assert(*frame_pointer < 128);
      score_ += kColorDistanceNTSC[(target_colors[i] * 128) + *frame_pointer];
    }
    // Sim output has pixels doubled horizontally.
    frame_pointer += 2;
  }
  score_valid_ = true;
}

size_t Kernel::OpcodeFieldIndex(size_t opcode_index) {
  size_t opcode_field = 0;
  while (opcode_index >= opcode_counts_[opcode_field])
    ++opcode_field;
  assert(opcode_field < opcode_counts_.size());
  return opcode_field;
}

void Kernel::Mutate() {
  // Pick a field of opcodes at random.
  std::uniform_int_distribution<size_t>
    opcode_index_distro(0, total_dynamic_opcodes_ - 1);
  size_t opcode_index = opcode_index_distro(engine_);
  size_t opcode_field = OpcodeFieldIndex(opcode_index);
  size_t opcode_baseline =
    opcode_field == 0 ? 0 : opcode_counts_[opcode_field - 1];
  assert(opcodes_[opcode_field].size() > 0);
  assert(opcode_index >= opcode_baseline);

  // Either modify an opcode or swap two of them based on random bit.
  std::uniform_int_distribution<int> swap_or_modify(0, 1);
  if (opcodes_[opcode_field].size() > 1 && swap_or_modify(engine_)) {
    std::uniform_int_distribution<size_t>
      opcode_within_field(0, opcodes_[opcode_field].size() - 1);
    size_t index_1 = opcode_index - opcode_baseline;
    assert(index_1 < opcodes_[opcode_field].size());
    size_t index_2 = opcode_within_field(engine_);
    assert(index_2 < opcodes_[opcode_field].size());
    uint32 op_1 = opcodes_[opcode_field][index_1];
    opcodes_[opcode_field][index_1] = opcodes_[opcode_field][index_2];
    opcodes_[opcode_field][index_2] = op_1;
  } else {
    size_t index = opcode_index - opcode_baseline;
    assert(index < opcodes_[opcode_field].size());
    OpCode old_op = static_cast<OpCode>(
        opcodes_[opcode_field][index] & 0x000000ff);
    switch (old_op) {
      case LDA_Immediate:
      case LDX_Immediate:
      case LDY_Immediate:
        opcodes_[opcode_field][index] = GenerateRandomLoad();
        break;

      case STA_ZeroPage:
      case STX_ZeroPage:
      case STY_ZeroPage:
        opcodes_[opcode_field][index] = GenerateRandomStore();
        break;

      default:
        assert(false);
        break;
    }
  }
}

}  // namespace vcmsc
