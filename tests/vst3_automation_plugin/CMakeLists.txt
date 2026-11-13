cmake_minimum_required(VERSION 3.18)
project(PedalboardAutomationProbe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(SDK "${CMAKE_CURRENT_LIST_DIR}/../../JUCE/modules/juce_audio_processors/format_types/VST3_SDK")
add_library(AutomationProbe MODULE probe.cpp
  "${SDK}/pluginterfaces/base/funknown.cpp"
  "${SDK}/pluginterfaces/base/coreiids.cpp"
  "${SDK}/pluginterfaces/base/ustring.cpp"
  "${SDK}/public.sdk/source/vst/vstinitiids.cpp")
target_include_directories(AutomationProbe PRIVATE "${SDK}")
set_target_properties(AutomationProbe PROPERTIES PREFIX "" SUFFIX "")
if(APPLE)
  target_link_libraries(AutomationProbe PRIVATE "-framework CoreFoundation")
  set_target_properties(AutomationProbe PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/AutomationProbe.vst3/Contents/MacOS")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/AutomationProbe.vst3/Contents")
  configure_file(Info.plist.in "${CMAKE_BINARY_DIR}/AutomationProbe.vst3/Contents/Info.plist" COPYONLY)
elseif(WIN32)
  target_link_libraries(AutomationProbe PRIVATE ole32)
  set_target_properties(AutomationProbe PROPERTIES SUFFIX ".vst3"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
else()
  set_target_properties(AutomationProbe PROPERTIES SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/AutomationProbe.vst3/Contents/${CMAKE_SYSTEM_PROCESSOR}-linux")
endif()
