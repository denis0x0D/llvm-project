//mlir-vulkan-runner simple.mlir --shared-libs=/home/khalikov/llvm-project/build/lib/libvulkan-runtime-wrappers.so --entry-point-result=void

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_1(%arg0 : f32, %arg1 : memref<12xf32>) attributes {gpu.kernel} {
      gpu.return
    }
  }

  func @main() {
    %0 = constant 2.0 : f32
    %1 = alloc() : memref<12xf32>
    %cst = constant 1 : index
    call @printMemRegister() : () -> ()
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<12xf32>) -> ()
    return
  }
  func @printMemRegister()
}
