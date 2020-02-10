add_llvm_library(MLIRGPUtoVulkanCallsConversion
  ConvertLaunchFuncToVulkanCalls.cpp
  )

target_link_libraries(MLIRGPUtoVulkanCallsConversion
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRStandardOps
  MLIRStandardToSPIRVTransforms
  MLIRSupport
  MLIRTransforms
  MLIRSPIRVSerialization
  )
