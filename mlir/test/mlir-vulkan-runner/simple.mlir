//mlir-vulkan-runner simple.mlir --shared-libs=/home/khalikov/llvm-project/build/lib/libvulkan-runtime-wrappers.so --entry-point-result=void

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_1(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg3 : memref<8xf32>) attributes {gpu.kernel} {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = load %arg0[%0] : memref<8xf32>
      gpu.return
    }
  }

  func @main() {
    %arg0 = alloc() : memref<8xf32>
    %arg1 = alloc() : memref<8xf32>
    %arg2 = alloc() : memref<8xf32>
    
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %arg0, %arg1, %arg2) { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index, memref<8xf32>, memref<8xf32>, memref<8xf32>) -> ()
    return
  }
}
