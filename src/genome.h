#ifndef SRC_GENOME_H_
#define SRC_GENOME_H_

#include <array>

#include "codon.h"
#include "kernel.h"
#include "spec.h"
#include "tls_prng.h"
#include "types.h"

namespace vcsmc {

// A Genome represents the genetic structure of an entire frame program. It
// wraps an array of Codons which are Translated in order, along with a
// a provided Spec, to produce a Kernel. Genomes are uniquely identifiable by
// their fingerprint, which is a hash computed across the Codons.
class Genome {
 public:
  Genome();
  Genome(const Genome&);

  // Populate the entire Genome with randomly-selected Codons.
  void GenerateRandom(TlsPrngList::reference tls_prng);

  // Given the supplied SpecList convert the Genome into a valid VCS program,
  // returning the results as a Kernel.
  Kernel Translate(const SpecList specs) const;

  uint64 fingerprint() const { return fingerprint_; }

 private:
  std::array<Codon, kFrameSizeCodons> codons_;
  uint64 fingerprint_;
};

}  // namespace vcsmc

#endif  // SRC_GENOME_H_
