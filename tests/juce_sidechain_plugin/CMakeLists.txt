cmake_minimum_required(VERSION 3.18)
project(PedalboardJUCESidechainAcceptance VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../JUCE" "${CMAKE_BINARY_DIR}/JUCE")
option(PEDALBOARD_BUILD_JUCE_SIDECHAIN_AU "Build AU wrapper (requires legacy Rez SDK resources)" OFF)
set(FORMATS VST3)
if(APPLE AND PEDALBOARD_BUILD_JUCE_SIDECHAIN_AU)
  list(APPEND FORMATS AU)
endif()
juce_add_plugin(SidechainAcceptance
  COMPANY_NAME "OpenMIRLab"
  BUNDLE_ID "org.openmirlab.sidechain-acceptance"
  PLUGIN_MANUFACTURER_CODE OMrl PLUGIN_CODE OmSc
  PRODUCT_NAME "JUCE NoiseGate Acceptance"
  FORMATS ${FORMATS} COPY_PLUGIN_AFTER_BUILD FALSE)
target_sources(SidechainAcceptance PRIVATE plugin.cpp)
target_compile_definitions(SidechainAcceptance PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(SidechainAcceptance PRIVATE juce::juce_audio_utils)
juce_add_console_app(NativeSidechainHost PRODUCT_NAME "Native Sidechain Control")
target_sources(NativeSidechainHost PRIVATE host.cpp)
target_compile_definitions(NativeSidechainHost PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_PLUGINHOST_VST3=1 JUCE_PLUGINHOST_AU=$<BOOL:${APPLE}>)
target_link_libraries(NativeSidechainHost PRIVATE juce::juce_audio_utils)

option(PEDALBOARD_BUILD_JUCE_COMBINED "Build unchanged AudioPluginDemo combined-input acceptance" OFF)
if(PEDALBOARD_BUILD_JUCE_COMBINED)
  juce_add_plugin(CombinedAcceptance
    COMPANY_NAME "OpenMIRLab" BUNDLE_ID "org.openmirlab.combined-acceptance"
    PLUGIN_MANUFACTURER_CODE OMrl PLUGIN_CODE OmCm
    PRODUCT_NAME "JUCE Combined Acceptance"
    NEEDS_MIDI_INPUT TRUE NEEDS_MIDI_OUTPUT TRUE
    FORMATS VST3 COPY_PLUGIN_AFTER_BUILD FALSE)
  target_sources(CombinedAcceptance PRIVATE combined.cpp)
  target_compile_definitions(CombinedAcceptance PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
  target_link_libraries(CombinedAcceptance PRIVATE juce::juce_audio_utils)
endif()
