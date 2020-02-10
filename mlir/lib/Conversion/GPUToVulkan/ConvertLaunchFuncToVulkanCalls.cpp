//===- ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
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

static constexpr const char *kSetBinaryShader = "setBinaryShader";
static constexpr const char *kSetEntryPoint = "setEntryPoint";
static constexpr const char *kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char *kRunOnVulkan = "runOnVulkan";
static constexpr const char *kSPIRVBinary = "SPIRV_BIN";

namespace {

class GpuLaunchFuncToVulkanCalssPass
    : public ModulePass<GpuLaunchFuncToVulkanCalssPass> {
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

  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);
  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);
  Value createEntryPointNameConstant(StringRef name, Location loc,
                                     OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  void runOnModule() override {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    initializeCachedTypes();

    getModule().walk(
        [this](mlir::gpu::LaunchFuncOp op) { translateGpuLaunchCalls(op); });

    // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
    for (auto gpuModule :
         llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
      gpuModule.erase();

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

void GpuLaunchFuncToVulkanCalssPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kSetEntryPoint)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetEntryPoint,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {getPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetNumWorkGroups)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetNumWorkGroups,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(), {getInt32Type(), getInt32Type(), getInt32Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetBinaryShader)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetBinaryShader,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {getPointerType(), getInt32Type()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kRunOnVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kRunOnVulkan,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {},
                                      /*isVarArg=*/false));
  }
}

Value GpuLaunchFuncToVulkanCalssPass::createEntryPointNameConstant(
    StringRef name, Location loc, OpBuilder &builder) {
  std::vector<char> shaderName(name.begin(), name.end());
  shaderName.push_back('\0');

  std::string globalName =
      std::string(llvm::formatv("{0}_spv_entry_point_name", name));
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(shaderName.data(), shaderName.size()),
      LLVM::Linkage::Internal, llvmDialect);
}

LogicalResult GpuLaunchFuncToVulkanCalssPass::createBinaryShader(
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

void GpuLaunchFuncToVulkanCalssPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(
          GpuLaunchFuncToVulkanCalssPass::createBinaryShader(module, binary))) {
    return signalPassFailure();
  }

  // Create call to `setBinaryShader` runtime function.
  Value ptrToSPIRVBinary = LLVM::createGlobalString(
      loc, builder, kSPIRVBinary, StringRef(binary.data(), binary.size()),
      LLVM::Linkage::Internal, getLLVMDialect());
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(binary.size()));
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetBinaryShader),
                               ArrayRef<Value>{ptrToSPIRVBinary, binarySize});

  // Create call to `setEntryPoint` runtime function.
  Value entryPointName =
      createEntryPointNameConstant(launchOp.kernel(), loc, builder);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetEntryPoint),
                               ArrayRef<Value>{entryPointName});

  // Create call to `setNumWorkGroups` runtime function.
  Value x = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(launchOp.getOperand(0)));
  Value y = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(launchOp.getOperand(1)));
  Value z = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(launchOp.getOperand(2)));
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetNumWorkGroups),
                               ArrayRef<Value>{x, y, z});

  // Create call to `runOnVulkan` runtime function.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kRunOnVulkan),
                               ArrayRef<Value>{});
  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToVulkanCallsPass() {
  return std::make_unique<GpuLaunchFuncToVulkanCalssPass>();
}

static PassRegistration<GpuLaunchFuncToVulkanCalssPass>
    pass("launch-func-to-vulkan",
         "Convert all launch_func ops to Vulkan runtime calls");
