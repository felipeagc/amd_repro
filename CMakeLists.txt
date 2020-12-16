cmake_minimum_required(VERSION 3.1)
project(repro VERSION 1.0 LANGUAGES C CXX)

add_executable(
	repro

	main.c

	glfw/glfw_unity.c

	rendergraph.c
	rendergraph_ext.c

	rendergraph.h
	rendergraph_ext.h
	spirv.h
	volk.h)

if(UNIX AND NOT APPLE)
	target_compile_options(repro PRIVATE -Wall -Wextra -Wno-unused-function)
	target_link_libraries(repro dl m pthread X11 Xau)
endif()
