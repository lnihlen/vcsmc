#ifndef SRC_PARTS_DIR_H_
#define SRC_PARTS_DIR_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class Dir : public Part {
 public:
  Dir(const std::string& name);
  virtual ~Dir() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_DIR_H_
