# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/masaya/PycharmProjects/mscf/mscf/lib/nuc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build

# Include any dependencies generated for this target.
include CMakeFiles/cnuc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cnuc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cnuc.dir/flags.make

CMakeFiles/cnuc.dir/nuc.cpp.o: CMakeFiles/cnuc.dir/flags.make
CMakeFiles/cnuc.dir/nuc.cpp.o: ../nuc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cnuc.dir/nuc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cnuc.dir/nuc.cpp.o -c /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/nuc.cpp

CMakeFiles/cnuc.dir/nuc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnuc.dir/nuc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/nuc.cpp > CMakeFiles/cnuc.dir/nuc.cpp.i

CMakeFiles/cnuc.dir/nuc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnuc.dir/nuc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/nuc.cpp -o CMakeFiles/cnuc.dir/nuc.cpp.s

CMakeFiles/cnuc.dir/nuc.cpp.o.requires:

.PHONY : CMakeFiles/cnuc.dir/nuc.cpp.o.requires

CMakeFiles/cnuc.dir/nuc.cpp.o.provides: CMakeFiles/cnuc.dir/nuc.cpp.o.requires
	$(MAKE) -f CMakeFiles/cnuc.dir/build.make CMakeFiles/cnuc.dir/nuc.cpp.o.provides.build
.PHONY : CMakeFiles/cnuc.dir/nuc.cpp.o.provides

CMakeFiles/cnuc.dir/nuc.cpp.o.provides.build: CMakeFiles/cnuc.dir/nuc.cpp.o


# Object files for target cnuc
cnuc_OBJECTS = \
"CMakeFiles/cnuc.dir/nuc.cpp.o"

# External object files for target cnuc
cnuc_EXTERNAL_OBJECTS =

/home/masaya/PycharmProjects/mscf/mscf/lib/libcnuc.so: CMakeFiles/cnuc.dir/nuc.cpp.o
/home/masaya/PycharmProjects/mscf/mscf/lib/libcnuc.so: CMakeFiles/cnuc.dir/build.make
/home/masaya/PycharmProjects/mscf/mscf/lib/libcnuc.so: CMakeFiles/cnuc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library /home/masaya/PycharmProjects/mscf/mscf/lib/libcnuc.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnuc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cnuc.dir/build: /home/masaya/PycharmProjects/mscf/mscf/lib/libcnuc.so

.PHONY : CMakeFiles/cnuc.dir/build

CMakeFiles/cnuc.dir/requires: CMakeFiles/cnuc.dir/nuc.cpp.o.requires

.PHONY : CMakeFiles/cnuc.dir/requires

CMakeFiles/cnuc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cnuc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cnuc.dir/clean

CMakeFiles/cnuc.dir/depend:
	cd /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/masaya/PycharmProjects/mscf/mscf/lib/nuc /home/masaya/PycharmProjects/mscf/mscf/lib/nuc /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build /home/masaya/PycharmProjects/mscf/mscf/lib/nuc/build/CMakeFiles/cnuc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cnuc.dir/depend

