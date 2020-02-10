set(LLVM_OPTIONAL_SOURCES
  ConvertGPUToSPIRV.cpp
  ConvertGPUToSPIRVPass.cpp
  ConvertLaunchFuncToVulkanCalls.cpp
  )

set(LLVM_TARGET_DEFINITIONS GPUToSPIRV.td)
mlir_tablegen(GPUToSPIRV.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToSPIRVIncGen)

add_llvm_library(MLIRGPUtoSPIRVTransforms
  ConvertGPUToSPIRV.cpp
  ConvertGPUToSPIRVPass.cpp
  )

add_dependencies(MLIRGPUtoSPIRVTransforms
  MLIRGPUToSPIRVIncGen)

set(LIBS
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRStandardOps
  MLIRStandardToSPIRVTransforms
  MLIRSupport
  MLIRTransforms
  )

target_link_libraries(MLIRGPUtoSPIRVTransforms ${LIBS})

add_llvm_library(MLIRGPUtoVulkanCallsConversion
  ConvertLaunchFuncToVulkanCalls.cpp
  )

target_link_libraries(MLIRGPUtoVulkanCallsConversion
  ${LIBS}
  MLIRSPIRVSerialization
  )
