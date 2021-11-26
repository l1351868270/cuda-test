cmake_minimum_required(VERSION 3.21)

project(cuda-test
  VERSION 0.0.1
  DESCRIPTION "a cuda test program."
  HOMEPAGE_URL "https://github.com/l1351868270/cuda-test.git"
  LANGUAGES C CXX CUDA)

add_executable(vec_add_h src/vec_add.c)
set_target_properties(vec_add_h PROPERTIES LINKER_LANGUAGE CXX)

add_executable(vec_add_d src/vec_add.cu)
add_executable(matrix_add src/matrix_add.cu)
add_executable(matrix_mul_h src/matrix_mul_h.c)
add_executable(matrix_mul_d src/matrix_mul_d.cu)
# set_target_properties(vec_add_d PROPERTIES LINKER_LANGUAGE CUDA)