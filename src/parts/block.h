#ifndef SRC_PARTS_BLOCK_H_
#define SRC_PARTS_BLOCK_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class Block1C61 : public Part {
 public:
  Block1C61(const std::string& name);
  virtual ~Block1C61() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

class Block1D41 : public Part {
 public:
  Block1D41(const std::string& name);
  virtual ~Block1D41() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

class Block4A81 : public Part {
 public:
  Block4A81(const std::string& name);
  virtual ~Block4A81() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

class Block4D41 : public Part {
 public:
  Block1D41(const std::string& name);
  virtual ~Block1D41() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_BLOCK_H_
