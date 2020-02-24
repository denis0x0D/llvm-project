add_llvm_library(MLIRGPUtoVulkanTransforms
  ConvertLaunchFuncToVulkanCalls.cpp
  ConvertGPULaunchFuncToVulkanLaunchFunc.cpp
  )

target_link_libraries(MLIRGPUtoVulkanTransforms
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  MLIRTranslation
  )
