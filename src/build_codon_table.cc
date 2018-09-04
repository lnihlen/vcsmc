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
    codons.push_back(
      vcsmc::MakeTIACodon(vcsmc::kSetPF0, static_cast<uint8>(i << 4)));
  }

  // kSetPF1
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetPF1, i));
  }

  // kSetPF2
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetPF2, i));
  }

  // kSetCTRLPF_REF - note that the don't care mask for each of the CTRLPF and
  // NUSIZ registers is set to include all bits impacted by the change.
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_REF, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_REF, 0x01));

  // kSetCTRLPF_SCORE
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_SCORE, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_SCORE, 0x02));

  // kSetCTRLPF_PFP
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_PFP, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_PFP, 0x04));

  // kSetCTRLPF_BALL (bits 4 and 5)
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_BALL, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_BALL, 0x10));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_BALL, 0x20));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCTRLPF_BALL, 0x30));

  // kSetNUSIZ0_P0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x01));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x02));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x03));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x04));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x05));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x06));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_P0, 0x07));

  // kSetNUSIZ0_M0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_M0, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_M0, 0x10));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_M0, 0x20));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ0_M0, 0x30));

  // kSetNUSIZ1_P1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x01));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x02));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x03));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x04));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x05));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x06));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_P1, 0x07));

  // kSetNUSIZ1_M1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_M1, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_M1, 0x10));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_M1, 0x20));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetNUSIZ1_M1, 0x30));

  // kStrobeRESP0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kStrobeRESP0, 0x00));

  // kStrobeRESP1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kStrobeRESP1, 0x00));

  // kStrobeRESM0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kStrobeRESM0, 0x00));

  // kStrobeRESM1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kStrobeRESM1, 0x00));

  // kStrobeRESBL
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kStrobeRESBL, 0x00));

  // kSetRESMP0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetRESMP0, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetRESMP0, 0x02));

  // kSetRESMP1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetRESMP1, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetRESMP1, 0x02));

  // kSetENAM0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetENAM0, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetENAM0, 0x02));

  // kSetENAM1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetENAM1, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetENAM1, 0x02));

  // kSetGRP0
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetGRP0, i));
  }

  // kSetGRP1
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetGRP1, i));
  }

  // kSetREFP0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetREFP0, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetREFP0, 0x08));

  // kSetREFP1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetREFP1, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetREFP1, 0x08));

  // kSetVDELP0
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetVDELP0, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetVDELP0, 0x01));

  // kSetVDELP1
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetVDELP1, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetVDELP1, 0x01));

  // kSetVDELBL
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetVDELBL, 0x00));
  codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetVDELBL, 0x01));

  // kSetCOLUP0
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCOLUP0, i));
  }

  // kSetCOLUP1
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCOLUP1, i));
  }

  // kSetCOLUPF
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCOLUPF, i));
  }

  // kSetCOLUBK
  for (auto i = 0; i < 256; ++i) {
    codons.push_back(vcsmc::MakeTIACodon(vcsmc::kSetCOLUBK, i));
  }

  // kWait
  for (auto i = 2; i < 256; ++i) {
    codons.push_back(vcsmc::MakeWaitCodon(i));
  }

  std::string code_path = FLAGS_output_directory + "/codon_table.cc";
  int fd = open(code_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
    S_IRUSR | S_IWUSR);
  if (fd < 0) {
    fprintf(stderr, "error creating code file %s.\n", code_path.c_str());
    return -1;
  }
  std::string code_top_string =
    "// File generated by src/build_codon_table.cc, edit there.\n"

    "#include \"codon.h\"\n"
    "#include \"codon_table.h\"\n\n"

    "namespace vcsmc {\n\n"

    "const Codon kCodonTable[kCodonTableSize] = {\n";
  size_t bytes_written = write(fd, code_top_string.c_str(),
    code_top_string.size());
  if (bytes_written != code_top_string.size()) {
    fprintf(stderr, "error writing code file %s.\n", code_path.c_str());
    return -1;
  }

  char buf[1024];
  size_t len;
  for (auto codon : codons) {
    len = snprintf(buf, 1024, "  0x%08x,\n", codon);
    write(fd, buf, len);
  }

  std::string code_bottom_string = "\n};\n\n}  // namespace vcsmc\n";
  write(fd, code_bottom_string.c_str(), code_bottom_string.size());

  close(fd);

  code_path = FLAGS_output_directory + "/codon_table.h";
  fd = open(code_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
    S_IRUSR | S_IWUSR);
  if (fd < 0) {
    fprintf(stderr, "error creating code file %s.\n", code_path.c_str());
    return -1;
  }
  len = snprintf(buf, 1024,
    "#ifndef OUT_CODON_TABLE_H_\n"
    "#define OUT_CODON_TABLE_H_\n\n"

    "#include \"types.h\"\n\n"

    "namespace vcsmc {\n\n"

    "const size_t kCodonTableSize = %lu;\n"
    "extern const uint32 kCodonTable[kCodonTableSize];\n\n"

    "}  // namespace vcsmc\n\n"
    "#endif  // OUT_CODON_TABLE_H_\n", codons.size());
    write(fd, buf, len);
    close(fd);

  return 0;
}
