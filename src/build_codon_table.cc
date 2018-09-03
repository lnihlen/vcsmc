// build_codon_table - generates codon_table.h/cc in the provided output
// directory. A complete table of all of supported Codons.

#include <fcntl.h>
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "codon.h"
#include "constants.h"
#include "types.h"

DEFINE_string(output_directory, "../out",
  "Output directory to save generated files to.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  std::vector<vcsmc::Codon> codons;

  // kSetPF0
  for (auto i = 0; i < 16; ++i) {
    // 4 MSb are used for PF0
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetPF0, vcsmc::PF0, static_cast<uint8>(i << 4), 0xf0));
  }

  // kSetPF1
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetPF1, vcsmc::PF1, static_cast<uint8>(i), 0xff));
  }

  // kSetPF2
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetPF2, vcsmc::PF2, static_cast<uint8>(i), 0xff));
  }

  // kSetCTRLPF_REF - note that the don't care mask for each of the CTRLPF and
  // NUSIZ registers is set to include all bits impacted by the change.
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_REF, vcsmc::CTRLPF, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_REF, vcsmc::CTRLPF, 0x01, 0x37));

  // kSetCTRLPF_SCORE
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_SCORE, vcsmc::CTRLPF, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_SCORE, vcsmc::CTRLPF, 0x02, 0x37));

  // kSetCTRLPF_PFP
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_PFP, vcsmc::CTRLPF, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_PFP, vcsmc::CTRLPF, 0x04, 0x37));

  // kSetCTRLPF_BALL (bits 4 and 5)
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_BALL, vcsmc::CTRLPF, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_BALL, vcsmc::CTRLPF, 0x10, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_BALL, vcsmc::CTRLPF, 0x20, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetCTRLPF_BALL, vcsmc::CTRLPF, 0x30, 0x37));

  // kSetNUSIZ0_P0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x01, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x02, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x03, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x04, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x05, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x06, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_P0, vcsmc::NUSIZ0, 0x07, 0x37));

  // kSetNUSIZ0_M0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_M0, vcsmc::NUSIZ0, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_M0, vcsmc::NUSIZ0, 0x10, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_M0, vcsmc::NUSIZ0, 0x20, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ0_M0, vcsmc::NUSIZ0, 0x30, 0x37));

  // kSetNUSIZ1_P1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x01, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x02, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x03, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x04, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x05, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x06, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_P1, vcsmc::NUSIZ1, 0x07, 0x37));

  // kSetNUSIZ1_M1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_M1, vcsmc::NUSIZ1, 0x00, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_M1, vcsmc::NUSIZ1, 0x10, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_M1, vcsmc::NUSIZ1, 0x20, 0x37));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetNUSIZ1_M1, vcsmc::NUSIZ1, 0x30, 0x37));

  // kStrobeRESP0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kStrobeRESP0, vcsmc::RESP0, 0x00, 0x00));

  // kStrobeRESP1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kStrobeRESP1, vcsmc::RESP1, 0x00, 0x00));

  // kStrobeRESM0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kStrobeRESM0, vcsmc::RESM0, 0x00, 0x00));

  // kStrobeRESM1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kStrobeRESM1, vcsmc::RESM1, 0x00, 0x00));

  // kStrobeRESBL
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kStrobeRESBL, vcsmc::RESBL, 0x00, 0x00));

  // kSetRESMP0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetRESMP0, vcsmc::RESMP0, 0x00, 0x02));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetRESMP0, vcsmc::RESMP0, 0x02, 0x02));

  // kSetRESMP1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetRESMP1, vcsmc::RESMP1, 0x00, 0x02));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetRESMP1, vcsmc::RESMP1, 0x02, 0x02));

  // kSetENAM0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetENAM0, vcsmc::ENAM0, 0x00, 0x02));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetENAM0, vcsmc::ENAM0, 0x02, 0x02));

  // kSetENAM1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetENAM1, vcsmc::ENAM1, 0x00, 0x02));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetENAM1, vcsmc::ENAM1, 0x02, 0x02));

  // kSetGRP0
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetGRP0, vcsmc::GRP0, static_cast<uint8>(i), 0xff));
  }

  // kSetGRP1
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetGRP1, vcsmc::GRP1, static_cast<uint8>(i), 0xff));
  }

  // kSetREFP0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetREFP0, vcsmc::REFP0, 0x00, 0x08));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetREFP0, vcsmc::REFP0, 0x08, 0x08));

  // kSetREFP1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetREFP1, vcsmc::REFP1, 0x00, 0x08));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetREFP1, vcsmc::REFP1, 0x08, 0x08));

  // kSetVDELP0
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetVDELP0, vcsmc::VDELP0, 0x00, 0x01));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetVDELP0, vcsmc::VDELP0, 0x01, 0x01));

  // kSetVDELP1
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetVDELP1, vcsmc::VDELP1, 0x00, 0x01));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetVDELP1, vcsmc::VDELP1, 0x01, 0x01));

  // kSetVDELBL
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetVDELBL, vcsmc::VDELBL, 0x00, 0x01));
  codons.push_back(
    vcsmc::MakeCodon(vcsmc::kSetVDELBL, vcsmc::VDELBL, 0x01, 0x01));

  // kSetCOLUP0
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetCOLUP0, vcsmc::COLUP0, static_cast<uint8>(i), 0xff));
  }

  // kSetCOLUP1
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetCOLUP1, vcsmc::COLUP1, static_cast<uint8>(i), 0xff));
  }

  // kSetCOLUPF
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetCOLUPF, vcsmc::COLUPF, static_cast<uint8>(i), 0xff));
  }

  // kSetCOLUBK
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kSetCOLUBK, vcsmc::COLUBK, static_cast<uint8>(i), 0xff));
  }

  // kWait
  for (auto i = 2; i < 256; ++i) {
    codons.push_back(vcsmc::MakeCodon(
      vcsmc::kWait, static_cast<uint8>(i), 0x00, 0xff));
  }

  std::string code_path = FLAGS_output_directory + "/codon_table.cc";
  int fd = open(code_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
    S_IRUSR | S_IWUSR);
  if (fd < 0) {
    fprintf(stderr, "error creating code file %s.\n", code_path.c_str());
    return -1;
  }
  std::string code_top_string =
    "// File generated by src/build_codon_table.cc, edit there.\n\n"

    "#include \"codon.h\"\n"
    "#include \"codon_table.h\"\n"

    "namespace vcsmc {\n\n";
  size_t bytes_written = write(fd, code_top_string.c_str(),
    code_top_string.size());
  if (bytes_written != code_top_string.size()) {
    fprintf(stderr, "error writing code file %s.\n", code_path.c_str());
    return -1;
  }

  char buf[1024];
  size_t len = snprintf(buf, 1024,
    "const size_t kCodonTableSize = %lu;\n\n"
    "const Codon kCodonTable[kCodonTableSize] = {\n", codons.size());
  write(fd, buf, len);

  for (auto codon : codons) {
    len = snprintf(buf, 1024, "\t0x%08x,\n", codon);
    write(fd, buf, len);
  }

  std::string code_bottom_string = "\n};\n\n}  // namespace vcsmc\n";
  write(fd, code_bottom_string.c_str(), code_bottom_string.size());

  close(fd);

  return 0;
}
