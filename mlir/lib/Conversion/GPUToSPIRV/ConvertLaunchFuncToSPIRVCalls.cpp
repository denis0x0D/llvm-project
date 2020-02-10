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

// To avoid name mangling, these are defined in the mini-runtime file.
static constexpr const char *cuModuleLoadName = "mcuModuleLoad";
static constexpr const char *cuModuleGetFunctionName = "mcuModuleGetFunction";
static constexpr const char *cuLaunchKernelName = "mcuLaunchKernel";
static constexpr const char *cuGetStreamHelperName = "mcuGetStreamHelper";
static constexpr const char *cuStreamSynchronizeName = "mcuStreamSynchronize";
static constexpr const char *kMcuMemHostRegister = "mcuMemHostRegister";
static const char *kBufferRegister = "bufferRegister";

static constexpr const char *kCubinAnnotation = "nvvm.cubin";
static constexpr const char *kCubinStorageSuffix = "_cubin_cst";

namespace {

class GpuLaunchFuncToSPIRVCallsPass
    : public ModulePass<GpuLaunchFuncToSPIRVCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    const llvm::Module &module = llvmDialect->getLLVMModule();
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmPointerPointerType = llvmPointerType.getPointerTo();
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
    llvmIntPtrType = LLVM::LLVMType::getIntNTy(
        llvmDialect, module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getVoidType() { return llvmVoidType; }

  LLVM::LLVMType getPointerType() { return llvmPointerType; }

  LLVM::LLVMType getPointerPointerType() { return llvmPointerPointerType; }

  LLVM::LLVMType getInt8Type() { return llvmInt8Type; }

  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  LLVM::LLVMType getIntPtrType() {
    const llvm::Module &module = getLLVMDialect()->getLLVMModule();
    return LLVM::LLVMType::getIntNTy(
        getLLVMDialect(), module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getCUResultType() {
    // This is declared as an enum in CUDA but helpers use i32.
    return getInt32Type();
  }

  // Allocate a void pointer on the stack.
  Value allocatePointer(OpBuilder &builder, Location loc) {
    auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                                builder.getI32IntegerAttr(1));
    return builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(), one,
                                          /*alignment=*/0);
  }

  void declareSPIRVFunctions(Location loc);
  Value setupParamsArray(gpu::LaunchFuncOp launchOp, OpBuilder &builder);
  Value generateKernelNameConstant(StringRef name, Location loc,
                                   OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  // Run the dialect converter on the module.
  void runOnModule() override {
    // Cache the LLVMDialect for the current module.
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    // Cache the used LLVM types.
    initializeCachedTypes();

    getModule().walk([this](mlir::gpu::LaunchFuncOp op) {
      translateGpuLaunchCalls(op);
    });

    // GPU kernel modules are no longer necessary since we have a global
    // constant with the CUBIN data.
    for (auto m :
         llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
      m.erase();

    // Erase spirv::Module operations.
    for (auto spirvModule :
         llvm::make_early_inc_range(getModule().getOps<spirv::ModuleOp>()))
      spirvModule.erase();
  }

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmPointerPointerType;
  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmIntPtrType;
};

} // anonymous namespace

// Adds declarations for the needed helper functions from the CUDA wrapper.
// The types in comments give the actual types expected/returned but the API
// uses void pointers. This is fine as they have the same linkage in C.
void GpuLaunchFuncToSPIRVCallsPass::declareSPIRVFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());
  if (!module.lookupSymbol(cuModuleLoadName)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, cuModuleLoadName,
        LLVM::LLVMType::getFunctionTy(
            getCUResultType(),
            {
                getPointerPointerType(), /* CUmodule *module */
                getPointerType()         /* void *cubin */
            },
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

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the 'nvvm.cubin' attribute of the
// kernel function in the IR.
void GpuLaunchFuncToSPIRVCallsPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder funcBuilder(module.getBody()->getTerminator());
  OpBuilder builder(launchOp);

  Location loc = launchOp.getLoc();

  bool done = false;
  SmallVector<uint32_t, 0> binary;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (done) {
      return signalPassFailure();
    }
    done = true;
    if (failed(spirv::serialize(spirvModule, binary))) {
      return signalPassFailure();
    }
  }

  std::vector<char> binaryShader;
  binaryShader.resize(binary.size() * sizeof(uint32_t));
  std::memcpy(binaryShader.data(), reinterpret_cast<char *>(binary.data()),
              binaryShader.size());

  Value ptrToExec = LLVM::createGlobalString(
      loc, builder, "executable",
      StringRef(binaryShader.data(), binaryShader.size()),
      LLVM::Linkage::Internal, getLLVMDialect());

  auto binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(binaryShader.size()));

  auto kernelName = generateKernelNameConstant(launchOp.kernel(), loc, builder);

  // FIXME: Add size value.
  funcBuilder.create<LLVM::LLVMFuncOp>(
      loc, "setEntryPoint",
      LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                    /*isVarArg=*/false));

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr("setEntryPoint"),
                               ArrayRef<Value>{kernelName});

  // FIXME: Add size value.
  funcBuilder.create<LLVM::LLVMFuncOp>(
      loc, "setBinaryShader",
      LLVM::LLVMType::getFunctionTy(llvmVoidType,
                                    {llvmPointerType, llvmInt32Type},
                                    /*isVarArg=*/false));

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr("setBinaryShader"),
                               ArrayRef<Value>{ptrToExec, binarySize});

  funcBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runOnVulkan",
      LLVM::LLVMType::getFunctionTy(llvmVoidType, {},
                                    /*isVarArg=*/false));

  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr("runOnVulkan"),
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
