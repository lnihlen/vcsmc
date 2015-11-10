// asm - super-limited 6502 assembler, supports arbitrarily long file sizes and
// only 3 opcodes - load immediate, store zero page, and nop.

#include <fcntl.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "assembler.h"
#include "opcode.h"
#include "types.h"

static const size_t kFileBufferSize = 16384 * 1024;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "asm usage: asm input_file.asm output_file.bin\n");
    return -1;
  }

  int input_fd = open(argv[1], O_RDONLY);
  if (input_fd < 0)
    return -1;

  int output_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR);
  if (output_fd < 0)
    return -1;

  vcsmc::Assembler::InitAssemblerTables();
  std::unique_ptr<char[]> input_buffer(new char[kFileBufferSize]);
  std::unique_ptr<uint8[]> output_buffer(new uint8[kFileBufferSize]);

  int bytes_read = read(input_fd, input_buffer.get(), kFileBufferSize);
  std::string line_fragment;
  while (bytes_read > 0) {
    std::vector<std::unique_ptr<vcsmc::op::OpCode>> opcodes;
    // Assemble a whole line from any fragment remaining from prior chunk of
    // data loaded.
    const char* chunk_start = input_buffer.get();
    if (line_fragment.length()) {
      size_t line_end = 0;
      for (; line_end < bytes_read; ++line_end) {
        if (input_buffer[line_end] == '\n')
          break;
      }
      // TODO: this is *not* working on my laptop right now, for some reason,
      // so I bump the buffer size up so it won't be a problem.
      line_fragment += std::string(chunk_start, line_end);
      if (!vcsmc::Assembler::AssembleString(line_fragment, &opcodes)) {
        fprintf(stderr, "error assembling fragment: %s\n",
            line_fragment.c_str());
        break;
      }
      if (line_end >= bytes_read)
        break;
      chunk_start += line_end;
    }

    // Back-scan to remove any line fragment from end of buffer (should only be
    // necessary if bytes_read == kFileBufferSize) and save in line_fragment.
    int chunk_end = bytes_read;
    if (bytes_read == kFileBufferSize) {
      for (; chunk_end >= 0; --chunk_end) {
        if (input_buffer[chunk_end] == '\n')
          break;
      }
      line_fragment = std::string(input_buffer.get() + chunk_end + 1,
          bytes_read - chunk_end);
    }

    // Pass giant substring to assembler with fresh output opcode vector. If
    // this fails we break out of the loop.
    const std::string chunk(chunk_start, chunk_end);
    if (!vcsmc::Assembler::AssembleString(chunk, &opcodes)) {
      fprintf(stderr, "error assembling chunk: %s\n", chunk.c_str());
      break;
    }

    // Traverse vector generating bytecode from each opcode to fill output
    // buffer.
    uint8* out_buffer_end = output_buffer.get();
    for (int i = 0; i < opcodes.size(); ++i)
      out_buffer_end += opcodes[i]->bytecode(out_buffer_end);

    // Write output buffer data to file.
    write(output_fd, output_buffer.get(), out_buffer_end - output_buffer.get());

    // Reset bytes_read by reading more of the input file.
    bytes_read = read(input_fd, input_buffer.get(), kFileBufferSize);
  }

  close(input_fd);
  close(output_fd);
  return 0;
}
