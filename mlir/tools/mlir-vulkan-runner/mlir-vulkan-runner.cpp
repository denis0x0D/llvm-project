//===- mlir-vulkan-runner.cpp - MLIR Vulkan Execution Driver---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Passes.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/JitRunner.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;
static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init(""));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  pm.addPass(createGpuKernelOutliningPass());
  // Lower host part to LLVM dialect.
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  // TODO: Handle work group size.
  pm.addPass(createConvertGPUToSPIRVPass({2, 2}));
  OpPassManager &modulePM = pm.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(createConvertGpuLaunchFuncToSPIRVCallsPass());
  return pm.run(m);
}
int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR Vulkan execution driver\n");
  mlir::registerPassManagerCLOptions();
  std::string errorMessage;
  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (!inputFilename.empty()) {
    auto inputFile = openInputFile(inputFilename, &errorMessage);
    if (!inputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());

    MLIRContext context;
    OwningModuleRef moduleRef(parseSourceFile(sourceMgr, &context));
    if (!moduleRef) {
      llvm::errs() << "\ncan not open the file" << '\n';
      return 1;
    }
    runMLIRPasses(moduleRef.get());
    moduleRef->print(outputFile->os());
  }
  return 0;
}

/*
int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
*/
