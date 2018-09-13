#include "genome.h"

#include "codon_table.h"
#include "snippet.h"
#include "state.h"


namespace vcsmc {

void Genome::GenerateRandom(TlsPrngList::reference tls_prng) {
  std::uniform_int_distribution<int> codon_distro(0, kCodonTableSize);

  // Populate entire genome with random lookups from table of all Codons.
  for (size_t i = 0; i < kFrameSizeCodons; ++i) {
    codons_[i] = kCodonTable[codon_distro(tls_prng)];
  }
}

const uint8* Genome::Sequence(const SpecList specs) {
  bytecode_size_ = 0;

  State state;
  size_t codon_index = 0;
  SpecIterator spec_iterator = specs->cbegin();

  while (state.current_time() < kScreenSizeCycles) {
    Snippet snippet = SequenceNext(state,
                                   codon_index,
                                   spec_iterator);
    state.Apply(snippet);
    std::memcpy(bytecode_.data() + bytecode_size_, snippet.bytecode.data(),
        snippet.size);
    bytecode_size_ += snippet.size;
  }
}

Snippet Genome::SequenceNext(const State& state,
                             size_t& codon_index,
                             SpecIterator& spec_iterator) {
  // Priorities are implementing specs on time, not overflowing a bank, and
  // then sequencing the Codons. First early-out for an exact match of timing
  // for Spec.
}




}  // namespace vcsmc
