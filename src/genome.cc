#include "genome.h"

#include <cstring>

#include "codon_table.h"
#include "snippet.h"
#include "state.h"

namespace vcsmc {

Genome::Genome()
  : bytecode_size_(0),
    first_untranslated_codon_(0) {
  condons_.fill(0);
  bytecode_.fill(0);
}

Genome::Genome(const Genome & genome)
  : bytecode_size_(0),
    first_untranslated_codon_(genome.first_untranslated_codon_) {
  std::memcpy(codons_.data(), genome.codons_.data(), sizeof(codons_));
  // We avoid copying the bytecode, as we assume it will be re-generated
  // after some mutation.
  bytecode_.fill(0);
}

void Genome::GenerateRandom(TlsPrngList::reference tls_prng) {
  std::uniform_int_distribution<int> codon_distro(0, kCodonTableSize);

  // Populate entire genome with random lookups from table of all Codons.
  for (size_t i = 0; i < kFrameSizeCodons; ++i) {
    codons_[i] = kCodonTable[codon_distro(tls_prng)];
  }
}

const uint8* Genome::Translate(const SpecList specs) {
  bytecode_size_ = 0;

  State state;
  size_t codon_index = 0;
  SpecIterator next_spec = specs->cbegin();
  bool translating_spec = false;
  size_t spec_codon_index = 0;
  Codon next_codon;

  while (state.current_time() < kScreenSizeCycles) {
    // If we are currently sequencing a Spec just continue with picking Codons
    // from that Spec.
    if (translating_spec) {
      if (spec_codon_index < next_spec->size()) {
        next_codon = next_spec->codons()[spec_codon_index];
        ++spec_codon_index;
      } else {
        translating_spec = false;
        ++next_spec;
      }
    }

    uint32 next_spec_time = next_spec != specs->cend() ?
        next_spec->start_time() : kScreenSizeCycles;

    // If next spec is due apply that.
    if (!translating_spec && state.current_time() >= next_spec_time) {
      // Zero-length Specs not supported.
      assert(next_spec->size() > 0);
      translating_spec = true;
      next_codon = *next_spec->codons();
      spec_codon_index = 1;
    }

    // OK, after checking that we are either in a spec already or should be in
    // one, if none of those cases apply we can source a Codon from the Genome.
    if (!translating_spec) {
      next_codon = codons_[codon_index];
      ++codon_index;
    }

    // Translate the supplied Codon.
    Snippet codon_snippet = state.Translate(next_codon);

    // Check for if we need to insert a bank switching Codon first.
    if (((bytecode_size_ + codon_snippet.size) % kBankSize) >
        (kBankSize - kBankPadding)) {
      Codon jump_codon = MakeBankSwitchCodon(bytecode_size_ % kBankSize);
      Snippet bank_snippet = state.Translate(jump_codon);
      state.Apply(bank_snippet, bytecode_.data() + bytecode_size_);
      bytecode_size_ += bank_snippet.size;
      // Note: it is assumed here that next_codon does not need re-translation.
    }

    state.Apply(codon_snippet, bytecode_.data() + bytecode_size_);
    bytecode_size_ += codon_snippet.size;
  }

  // We store the final
  last_translated_codon_ = codon_index;

  return bytecode_.data();
}

}  // namespace vcsmc
