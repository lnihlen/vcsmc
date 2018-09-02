#ifndef SRC_GENOME_H_
#define SRC_GENOME_H_

#include "codon.h"
#include "spec.h"
#include "tls_prng.h"

namespace vcsmc {

class Genome {
 public:
  // Populate the entire Genome with randomly-selected Codons.
  void GenerateRandom(TlsPrngList::reference tls_prng);
  // Given the supplied specs convert the Genome into a valid VCS program,
  // saves the result into |bytecode_|.
  const uint8* Sequence(const SpecList specs);

  const uint8* bytecode() { return bytecode_.get(); }

 private:
  std::array<Codon, kFrameSizeCodons> codons_;
  std::unique_ptr<uint8> bytecode_;
};

}  // namespace vcsmc

#endif  // SRC_GENOME_H_
