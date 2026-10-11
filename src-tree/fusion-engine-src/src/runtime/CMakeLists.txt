# The measured device route profiles ship inside the binary: device_profiles.json (ninfer-calibrate
# --print output of each part, merged) becomes a string in the generated source.
set(NINFER_DEVICE_PROFILES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/engine/device_profiles.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NINFER_DEVICE_PROFILES_FILE})
file(READ ${NINFER_DEVICE_PROFILES_FILE} NINFER_DEVICE_PROFILES_JSON)
if(MSVC)
  # MSVC 19.44 limits each literal to 16,380 characters. Adjacent raw literals
  # retain the JSON bytes while allowing the bundled profiles to grow past that limit.
  set(_profile_json_remaining "${NINFER_DEVICE_PROFILES_JSON}")
  set(NINFER_DEVICE_PROFILES_JSON "")
  string(LENGTH "${_profile_json_remaining}" _profile_json_length)
  while(_profile_json_length GREATER 16000)
    string(SUBSTRING "${_profile_json_remaining}" 0 16000 _profile_json_chunk)
    string(APPEND NINFER_DEVICE_PROFILES_JSON "${_profile_json_chunk})ninfer_json\" R\"ninfer_json(")
    string(SUBSTRING "${_profile_json_remaining}" 16000 -1 _profile_json_remaining)
    string(LENGTH "${_profile_json_remaining}" _profile_json_length)
  endwhile()
  string(APPEND NINFER_DEVICE_PROFILES_JSON "${_profile_json_remaining}")
endif()
configure_file(engine/device_profiles_builtin.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/engine/device_profiles_builtin.cpp @ONLY)

add_library(ninfer_runtime_support STATIC
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/device_profile.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/engine/device_profiles_builtin.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
  prefix_cache/block_hash.cpp
  prefix_cache/prefix_index.cpp
  prefix_cache/tap_planner.cpp
)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core ninfer_ops PRIVATE ninfer::json)

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninfer_engine STATIC
  engine/model_instance.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_engine PRIVATE
  ninfer_calibration
  ninfer_model_runtime
  ninfer_artifact
  ninfer_core
  ninfer_runtime_support
  ninfer_ops
  ninfer_text
  ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)
