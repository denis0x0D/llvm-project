//===- ConvertLaunchFuncToSPIRVCalls.cpp - MLIR SPIRV lowering passes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

static constexpr const char *kSetEntryPoint = "setEntryPoint";
static constexpr const char *kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char *kSetBinaryShader = "setBinaryShader";
static constexpr const char *kRunOnVulkan = "runOnVulkan";
static constexpr const char *kSPIRV_BIN = "SPIRV_BIN";

namespace {

class GpuLaunchFuncToSPIRVCallsPass
    : public ModulePass<GpuLaunchFuncToSPIRVCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
  }

  LLVM::LLVMType getVoidType() { return llvmVoidType; }
  LLVM::LLVMType getPointerType() { return llvmPointerType; }
  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  void declareVulkanFunctions(Location loc);
  Value generateKernelNameConstant(StringRef name, Location loc,
                                   OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);
  LogicalResult generateBinaryShader(ModuleOp module,
                                     std::vector<char> &binaryShader);

public:
  void runOnModule() override {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    initializeCachedTypes();

    getModule().walk([this](mlir::gpu::LaunchFuncOp op) {
      translateGpuLaunchCalls(op);
    });

    // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
    for (auto m :
         llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
      m.erase();

    for (auto spirvModule :
         llvm::make_early_inc_range(getModule().getOps<spirv::ModuleOp>()))
      spirvModule.erase();
  }

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
};
} // anonymous namespace

void GpuLaunchFuncToSPIRVCallsPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kSetEntryPoint)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetEntryPoint,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetNumWorkGroups)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetNumWorkGroups,
        LLVM::LLVMType::getFunctionTy(
            llvmVoidType, {llvmInt32Type, llvmInt32Type, llvmInt32Type},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetBinaryShader)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetBinaryShader,
        LLVM::LLVMType::getFunctionTy(llvmVoidType,
                                      {llvmPointerType, llvmInt32Type},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kRunOnVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kRunOnVulkan,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {},
                                      /*isVarArg=*/false));
  }
}

Value GpuLaunchFuncToSPIRVCallsPass::generateKernelNameConstant(
    StringRef name, Location loc, OpBuilder &builder) {
  // Make sure the trailing zero is included in the constant.
  std::vector<char> kernelName(name.begin(), name.end());
  kernelName.push_back('\0');

  std::string globalName = std::string(llvm::formatv("{0}_kernel_name", name));
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(kernelName.data(), kernelName.size()),
      LLVM::Linkage::Internal, llvmDialect);
}

LogicalResult GpuLaunchFuncToSPIRVCallsPass::generateBinaryShader(
    ModuleOp module, std::vector<char> &binaryShader) {
  bool done = false;
  SmallVector<uint32_t, 0> binary;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (done) {
      return failure();
    }
    done = true;
    if (failed(spirv::serialize(spirvModule, binary))) {
      return failure();
    }
  }
  binaryShader.resize(binary.size() * sizeof(uint32_t));
  std::memcpy(binaryShader.data(), reinterpret_cast<char *>(binary.data()),
              binaryShader.size());

  return success();
}

void GpuLaunchFuncToSPIRVCallsPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();
  
  declareVulkanFunctions(loc);
  std::vector<char> binaryShader;

  if (failed(GpuLaunchFuncToSPIRVCallsPass::generateBinaryShader(
          module, binaryShader))) {
    return signalPassFailure();
  }

  Value ptrToExec = LLVM::createGlobalString(
      loc, builder, kSPIRV_BIN,
      StringRef(binaryShader.data(), binaryShader.size()),
      LLVM::Linkage::Internal, getLLVMDialect());

  auto binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(binaryShader.size()));

  auto kernelName = generateKernelNameConstant(launchOp.kernel(), loc, builder);

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr(kSetEntryPoint),
                               ArrayRef<Value>{kernelName});

  Value x = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(launchOp.getOperand(0)));
  Value y = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(launchOp.getOperand(1)));
  Value z = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(launchOp.getOperand(2)));

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr(kSetNumWorkGroups),
                               ArrayRef<Value>{x, y, z});

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr(kSetBinaryShader),
                               ArrayRef<Value>{ptrToExec, binarySize});

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr(kRunOnVulkan),
                               ArrayRef<Value>{});
  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToSPIRVCallsPass() {
  return std::make_unique<GpuLaunchFuncToSPIRVCallsPass>();
}

static PassRegistration<GpuLaunchFuncToSPIRVCallsPass>
    pass("launch-func-to-spirv",
         "Convert all launch_func ops to SPIRV runtime calls");
