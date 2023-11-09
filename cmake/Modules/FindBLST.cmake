# Findblst.cmake
find_path(BLST_INCLUDE_DIR NAMES blst)
find_library(BLST_LIBRARY NAMES blst)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blst DEFAULT_MSG BLST_LIBRARY BLST_INCLUDE_DIR)

if(BLST_FOUND AND NOT TARGET blst::blst)
  add_library(blst::blst UNKNOWN IMPORTED)
  set_target_properties(blst::blst PROPERTIES
    IMPORTED_LOCATION "${BLST_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${BLST_INCLUDE_DIR}")
endif()