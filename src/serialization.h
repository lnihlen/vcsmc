#ifndef SRC_SERIALIZATION_H_
#define SRC_SERIALIZATION_H_

#include <memory>
#include <vector>

#include "spec.h"

namespace vcsmc {

// Encodes the provided bytes into a longer Base64 string with line breaks on
// 80 characters and indented with spaces to |indent|, if |indent| is nonzero,
// or just plain string if |indent| is zero.
std::string Base64Encode(const uint8* bytes, size_t size, size_t indent);

// It is assumed string is already concatenated for the decode.
std::unique_ptr<uint8[]> Base64Decode(const std::string& base64, size_t size);

}  // namespace vcsmc

#endif  // SRC_SERIALIZATION_H_
