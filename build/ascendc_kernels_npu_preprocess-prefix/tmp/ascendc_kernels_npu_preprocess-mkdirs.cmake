# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/fanyi/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/device_preprocess_project"
  "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/src/ascendc_kernels_npu_preprocess-build"
  "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix"
  "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/tmp"
  "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/src/ascendc_kernels_npu_preprocess-stamp"
  "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/src"
  "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/src/ascendc_kernels_npu_preprocess-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/src/ascendc_kernels_npu_preprocess-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/fanyi/fyfy/build/ascendc_kernels_npu_preprocess-prefix/src/ascendc_kernels_npu_preprocess-stamp${cfgdir}") # cfgdir has leading slash
endif()
