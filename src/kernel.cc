#include "kernel.h"

#include <array>
#include <cassert>
#include <cstring>
#include <random>

#include "assembler.h"

namespace vcsmc {

Kernel::Kernel(std::seed_seq& seed)
  : engine_(seed),
    specs_(new std::vector<Spec>()),
    bytecode_size_(0),
    fingerprint_(0) {
}

void Kernel::GenerateRandomKernelJob::Execute() {
  uint32 current_cycle = 0;
  size_t spec_list_index = 0;
  size_t total_byte_size = 0;
  // We reserve 3 cycles at the end for the jmp instruction taking us back to
  // the top.
  while (current_cycle < kScreenSizeCycles - 3) {
    uint32 next_spec_start_time = spec_list_index < specs_->size() ?
        specs_->at(spec_list_index).range().start_time() : kInfinity;
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
      uint32_t starting_cycle = current_cycle;
      std::unique_ptr<std::vector<uint32>> ops(new std::vector<uint32>());
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
        if (bytes_remaining < kBankPadding ||
            (bytes_remaining < next_spec_size &&
                cycles_remaining < kBankPadding)) {
          kernel_->opcodes_.emplace_back(std::move(ops));
          ops.reset(new std::vector<uint32>());
          kernel_->opcode_ranges_.emplace_back(starting_cycle, current_cycle);
          kernel_->AppendJmpSpec(current_cycle, total_byte_size % kBankSize);
          const Spec& jmp_spec = kernel_->specs_->back();
          current_cycle += jmp_spec.range().Duration();
          assert(cycles_remaining > jmp_spec.range().Duration());
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
          ops->push_back(op);
        }
      }
      assert(ops->size());
      kernel_->opcodes_.emplace_back(std::move(ops));
      kernel_->opcode_ranges_.emplace_back(starting_cycle, current_cycle);
    }
  }
  assert(current_cycle == kScreenSizeCycles - 3);
  kernel_->AppendJmpSpec(current_cycle, total_byte_size % kBankSize);
  kernel_->RegenerateBytecode(total_byte_size);
}

uint32 Kernel::GenerateRandomOpcode(uint32 cycles_remaining) {
  // As no possible instruction lasts only one cycle we cannot support a
  // situation where only one cycle is remaining.
  assert(cycles_remaining > 1);
  // With two or four cycles remaining we must only generate two-cycle
  // instructions.
  if (cycles_remaining == 2 || cycles_remaining == 4) {
    std::uniform_int_distribution<int> distro(0, 3);
    if (distro(engine_) == 0) {
      return PackOpCode(OpCode::NOP_Implied, 0, 0);
    } else {
      return GenerateRandomLoad();
    }
  } else if (cycles_remaining == 3) {
    return GenerateRandomStore();
  }

  std::uniform_int_distribution<int> distro(0, 6);
  int roll = distro(engine_);
  if (roll == 0) {
    return PackOpCode(OpCode::NOP_Implied, 0, 0);
  } else if (roll <= 3) {
    return GenerateRandomLoad();
  }
  return GenerateRandomStore();
}

uint32 Kernel::GenerateRandomLoad() {
  std::uniform_int_distribution<uint8> arg_distro(0, 255);
  uint8 arg = arg_distro(engine_);
  std::uniform_int_distribution<int> op_distro(0, 2);
  switch (op_distro(engine_)) {
    case 0:
      return PackOpCode(OpCode::LDA_Immediate, arg, 0);

    case 1:
      return PackOpCode(OpCode::LDX_Immediate, arg, 0);

    case 2:
      return PackOpCode(OpCode::LDY_Immediate, arg, 0);

    default:
      assert(false);
      break;
  }
  return 0;
}

uint32 Kernel::GenerateRandomStore() {
  std::array<uint8, 34> kValidTIA{{
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
    TIA::HMP0,
    TIA::HMP1,
    TIA::HMM0,
    TIA::HMM1,
    TIA::HMBL,
    TIA::VDELP0,
    TIA::VDELP1,
    TIA::VDELBL,
    TIA::RESMP0,
    TIA::RESMP1,
    TIA::HMOVE,
    TIA::HMCLR
  }};
  std::uniform_int_distribution<size_t> tia_distro(0, kValidTIA.size() - 1);
  uint8 tia = kValidTIA[tia_distro(engine_)];
  std::uniform_int_distribution<int> op_distro(0, 2);
  switch (op_distro(engine_)) {
    case 0:
      return PackOpCode(OpCode::STA_ZeroPage, tia, 0);

    case 1:
      return PackOpCode(OpCode::STX_ZeroPage, tia, 0);

    case 2:
      return PackOpCode(OpCode::STY_ZeroPage, tia, 0);

    default:
      assert(false);
      break;
  }
  return 0;
}

void Kernel::AppendJmpSpec(uint32 current_cycle, size_t current_bank_size) {
  assert(current_bank_size < (kBankSize - kBankPadding));
  size_t bank_balance_size = kBankSize - current_bank_size;
  std::unique_ptr<uint8[]> bytecode(new uint8[bank_balance_size]);
  std::memset(bytecode.get() + 3, 0, bank_balance_size - 3);
  bytecode_.get()[0] = JMP_Absolute;
  bytecode_.get()[1] = 0x00;
  bytecode_.get()[2] = 0xf0;
  specs_->emplace_back(
      Range(current_cycle, current_cycle + 3),
      bank_balance_size,
      std::move(bytecode));
}

void Kernel::RegenerateBytecode(size_t bytecode_size) {
  bytecode_size_ = bytecode_size;
  bytecode_.reset(new uint8[bytecode_size]);
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
      std::vector<uint32>* ops = opcodes_[current_range_index].get();
      // Serialize the packed opcodes into the buffer.
      for (size_t i = 0; i < ops->size(); ++i) {
        current_byte += UnpackOpCode(ops->at(i), current_byte);
        current_cycle += OpCodeCycles(ops->at(i));
      }
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
}

}  // namespace vcmsc
