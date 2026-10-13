# The CUDA MMQ consumer adds input addressing only. Never patch the shared GGML checkout.
# This overlay is tied to ggml 3cf03257; a changed donor must be reviewed rather than fuzzy-patched.
set(_strata_mmq_original "${_strata_mmq}")
set(_strata_mmq_patch "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/mmq-input-index.patch")
file(SHA256 "${_strata_mmq_original}/mmq.cuh" _strata_mmq_sha)
if(NOT _strata_mmq_sha STREQUAL "50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b")
  message(FATAL_ERROR "MMQ input-index overlay requires the pinned ggml 3cf03257 header")
endif()
find_program(STRATA_PATCH_EXECUTABLE NAMES patch REQUIRED)
set(_strata_mmq "${CMAKE_CURRENT_BINARY_DIR}/mmq-input-index")
file(MAKE_DIRECTORY "${_strata_mmq}")
file(COPY "${_strata_mmq_original}/" DESTINATION "${_strata_mmq}")
# file(COPY) may retain a newer patched header on reconfigure. Always start from the original bytes.
configure_file("${_strata_mmq_original}/mmq.cuh" "${_strata_mmq}/mmq.cuh" COPYONLY)
execute_process(COMMAND "${STRATA_PATCH_EXECUTABLE}" --batch --fuzz=0 -p0 -i "${_strata_mmq_patch}"
                WORKING_DIRECTORY "${_strata_mmq}" RESULT_VARIABLE _strata_mmq_patch_result)
if(NOT _strata_mmq_patch_result EQUAL 0)
  message(FATAL_ERROR "Cannot apply the pinned MMQ input-index patch")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${_strata_mmq_original}/mmq.cuh" "${_strata_mmq_patch}")
