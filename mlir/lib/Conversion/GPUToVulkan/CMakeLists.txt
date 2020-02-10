add_llvm_library(MLIRGPUtoVulkanCallsConversion
  ConvertLaunchFuncToVulkanCalls.cpp
  )

target_link_libraries(MLIRGPUtoVulkanCallsConversion
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  )
