set(LLVM_OPTIONAL_SOURCES
  ConvertGPUToSPIRV.cpp
  ConvertGPUToSPIRVPass.cpp
  ConvertLaunchFuncToSPIRVCalls.cpp
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

target_link_libraries(MLIRGPUtoSPIRVTransforms
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRStandardOps
  MLIRStandardToSPIRVTransforms
  MLIRSupport
  MLIRTransforms
  )

add_llvm_library(MLIRGPUtoSPIRVConversion
  ConvertLaunchFuncToSPIRVCalls.cpp
  )

add_dependencies(MLIRGPUtoSPIRVConversion
  MLIRGPUToSPIRVIncGen)

target_link_libraries(MLIRGPUtoSPIRVConversion
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRStandardToSPIRVTransforms
  MLIRSupport
  MLIRTransforms
  )
 
