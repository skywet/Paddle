# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif(APPLE)

add_subdirectory(details)

if(WITH_MKLDNN)
  set(mkldnn_quantizer_cfg mkldnn_quantizer_config)
  set(mkldnn_quantizer_src ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn_quantizer.cc)
  cc_library(${mkldnn_quantizer_cfg} SRCS mkldnn_quantizer_config.cc DEPS lod_tensor paddle_pass_builder)
  set(mkldnn_quantizer_cfg ${mkldnn_quantizer_cfg} PARENT_SCOPE)
endif()

cc_library(analysis_config SRCS analysis_config.cc DEPS ${mkldnn_quantizer_cfg} lod_tensor paddle_pass_builder table_printer)
cc_library(paddle_infer_contrib SRCS paddle_infer_contrib.cc DEPS zero_copy_tensor)
cc_library(paddle_pass_builder SRCS paddle_pass_builder.cc)

if(WITH_CRYPTO)
    cc_library(paddle_inference_api SRCS api.cc api_impl.cc helper.cc DEPS lod_tensor scope reset_tensor_array 
              analysis_config paddle_infer_contrib zero_copy_tensor trainer_desc_proto paddle_crypto custom_operator)
else()
    cc_library(paddle_inference_api SRCS api.cc api_impl.cc helper.cc DEPS lod_tensor scope reset_tensor_array 
              analysis_config paddle_infer_contrib zero_copy_tensor trainer_desc_proto custom_operator)
endif()

if(WIN32)
    target_link_libraries(paddle_inference_api gflags)
endif()

set(inference_deps ${analysis_deps} paddle_inference_api analysis naive_executor ${GLOB_PASS_LIB})

if(WITH_GPU AND TENSORRT_FOUND)
    set(inference_deps ${inference_deps} tensorrt_engine tensorrt_converter)
endif()

cc_library(analysis_predictor SRCS analysis_predictor.cc ${mkldnn_quantizer_src} DEPS ${inference_deps} 
          zero_copy_tensor ir_pass_manager op_compatible_info infer_io_utils)

cc_test(test_paddle_inference_api SRCS api_tester.cc DEPS paddle_inference_api)

if(WITH_TESTING)
  if (NOT APPLE AND NOT WIN32)
    inference_base_test(test_api_impl SRCS api_impl_tester.cc DEPS paddle_inference_shared
      ARGS --word2vec_dirname=${WORD2VEC_MODEL_DIR} --book_dirname=${IMG_CLS_RESNET_INSTALL_DIR})
  elseif(WIN32)
    inference_base_test(test_api_impl SRCS api_impl_tester.cc DEPS ${inference_deps}
      ARGS --word2vec_dirname=${WORD2VEC_MODEL_DIR} --book_dirname=${IMG_CLS_RESNET_INSTALL_DIR})
  endif()

endif()

if (NOT APPLE AND NOT WIN32)
  cc_test(test_analysis_predictor SRCS analysis_predictor_tester.cc DEPS paddle_inference_shared
          ARGS --dirname=${WORD2VEC_MODEL_DIR})
elseif (WIN32)
  cc_test(test_analysis_predictor SRCS analysis_predictor_tester.cc DEPS analysis_predictor benchmark ${inference_deps}
          ARGS --dirname=${WORD2VEC_MODEL_DIR})
endif()

if(WITH_TESTING AND WITH_MKLDNN)
  if (NOT APPLE AND NOT WIN32)
    cc_test(test_mkldnn_quantizer SRCS mkldnn_quantizer_tester.cc DEPS paddle_inference_shared ARGS --dirname=${WORD2VEC_MODEL_DIR})
  elseif (WIN32)
    cc_test(test_mkldnn_quantizer SRCS mkldnn_quantizer_tester.cc DEPS analysis_predictor benchmark ${inference_deps}
            ARGS --dirname=${WORD2VEC_MODEL_DIR})
  endif()
endif()

if(WITH_TESTING AND TEST test_api_impl)
    if(NOT APPLE)
        set_tests_properties(test_api_impl PROPERTIES TIMEOUT 120)
    endif()
endif()
