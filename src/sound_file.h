#ifndef SRC_SOUND_FILE_H_
#define SRC_SOUND_FILE_H_

#include <memory>
#include <string>

namespace vcsmc {

class Sound;

std::unique_ptr<Sound> LoadSound(const std::string& file_name);
bool SaveSound(const Sound* sound, const std::string& file_name);

}  // namespace vcsmc

#endif  // SRC_SOUND_FILE_H_
