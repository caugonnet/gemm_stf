# This file is loaded by CMake project() code injection hooks
# after the `project(CCCL)` call. That allows us to work around
# issues in the project CMakeLists.txt without having to patch
# the project

# Due to the fact that cccl/cudax/CMakeLists.txt doesn't early
# terminate when not TOPLEVEL we need to fix the cmake includes
# so that it can create the test targets
include("${PROJECT_SOURCE_DIR}/cmake/CCCLConfigureTarget.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/CCCLBuildCompilerTargets.cmake")

# Tell cudax that not to build any test logic.
# Again needed due to the missing early terminate
set(cudax_ENABLE_HEADER_TESTING OFF)
set(cudax_ENABLE_TESTING OFF)
set(cudax_ENABLE_EXAMPLES OFF)
