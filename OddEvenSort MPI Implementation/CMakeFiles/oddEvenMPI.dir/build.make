# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

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
CMAKE_COMMAND = /sw/eb/sw/CMake/3.12.1/bin/cmake

# The command to remove a file.
RM = /sw/eb/sw/CMake/3.12.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/user/erichaddad/MPI_Implementation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/user/erichaddad/MPI_Implementation

# Include any dependencies generated for this target.
include CMakeFiles/oddEvenMPI.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/oddEvenMPI.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/oddEvenMPI.dir/flags.make

CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.o: CMakeFiles/oddEvenMPI.dir/flags.make
CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.o: oddEvenMPI.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/user/erichaddad/MPI_Implementation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.o"
	/sw/eb/sw/GCCcore/10.2.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.o -c /scratch/user/erichaddad/MPI_Implementation/oddEvenMPI.cpp

CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.i"
	/sw/eb/sw/GCCcore/10.2.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/user/erichaddad/MPI_Implementation/oddEvenMPI.cpp > CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.i

CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.s"
	/sw/eb/sw/GCCcore/10.2.0/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/user/erichaddad/MPI_Implementation/oddEvenMPI.cpp -o CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.s

# Object files for target oddEvenMPI
oddEvenMPI_OBJECTS = \
"CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.o"

# External object files for target oddEvenMPI
oddEvenMPI_EXTERNAL_OBJECTS =

oddEvenMPI: CMakeFiles/oddEvenMPI.dir/oddEvenMPI.cpp.o
oddEvenMPI: CMakeFiles/oddEvenMPI.dir/build.make
oddEvenMPI: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
oddEvenMPI: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
oddEvenMPI: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
oddEvenMPI: /lib64/librt.so
oddEvenMPI: /lib64/libpthread.so
oddEvenMPI: /lib64/libdl.so
oddEvenMPI: CMakeFiles/oddEvenMPI.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/user/erichaddad/MPI_Implementation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable oddEvenMPI"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/oddEvenMPI.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/oddEvenMPI.dir/build: oddEvenMPI

.PHONY : CMakeFiles/oddEvenMPI.dir/build

CMakeFiles/oddEvenMPI.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/oddEvenMPI.dir/cmake_clean.cmake
.PHONY : CMakeFiles/oddEvenMPI.dir/clean

CMakeFiles/oddEvenMPI.dir/depend:
	cd /scratch/user/erichaddad/MPI_Implementation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/user/erichaddad/MPI_Implementation /scratch/user/erichaddad/MPI_Implementation /scratch/user/erichaddad/MPI_Implementation /scratch/user/erichaddad/MPI_Implementation /scratch/user/erichaddad/MPI_Implementation/CMakeFiles/oddEvenMPI.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/oddEvenMPI.dir/depend

