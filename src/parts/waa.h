#ifndef SRC_PARTS_WAA_H_
#define SRC_PARTS_WAA_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class WAA : public Part {
 public:
  WAA(const std::string& name);
  virtual ~WAA() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_WAA_H_
