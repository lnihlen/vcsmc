#include "genome.h"

#include "codon_table.h"

namespace vcsmc {

void Genome::GenerateRandom(TlsPrngList::reference tls_prng) {
  std::uniform_int_distribution<int> codon_distro(0, kCodonTableSize);

  // Populate entire genome with random lookups from table of all codons.
  for (size_t i = 0; i < kFrameSizeCodons; ++i) {
    codons_[i] = kCodonTable[codon_distro(tls_prng)];
  }
}

const uint8* Genome::Sequence(const SpecList specs) {
  std::vector<uint8> opcodes;
  size_t bank_size = 0;
  uint32 cycle_count;
  size_t codon_index = 0;
  size_t spec_list_index = 0;
  uint8 tia_state[TIA_COUNT];
  uint8 next_tia_state[TIA_COUNT];

  while (current_cycle < kScreenSizeCycles) {
  }

  return nullptr;
}

}  // namespace vcsmc
