#ifndef SRC_PARTS_DL_H_
#define SRC_PARTS_DL_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class Dl : public Part {
 public:
  Dl(const std::string& name);
  virtual ~Dl() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_DL_H_
