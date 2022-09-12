cmake_minimum_required(VERSION 3.10)
project(packed-fsm VERSION 0.0.1 LANGUAGES C)

add_executable(fsm-tables fsm-tables.c)

add_custom_command(
  OUTPUT transitions.h
  COMMAND fsm-tables
  ARGS transitions.h
  DEPENDS fsm.h fsm-tables.c)

add_executable(packed-fsm packed-fsm.c fsm.h transitions.h)
target_include_directories(packed-fsm PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_compile_options(packed-fsm PRIVATE "-msse4.1" "-mpclmul")
set_target_properties(packed-fsm PROPERTIES C_STANDARD_REQUIRED 11)
