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
CMAKE_COMMAND = /sw/eb/sw/CMake/3.12.1-GCCcore-7.3.0/bin/cmake

# The command to remove a file.
RM = /sw/eb/sw/CMake/3.12.1-GCCcore-7.3.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/scratch/user/erichaddad/Cuda Implementation"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/scratch/user/erichaddad/Cuda Implementation"

# Include any dependencies generated for this target.
include CMakeFiles/oddEvenSort.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/oddEvenSort.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/oddEvenSort.dir/flags.make

CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o: CMakeFiles/oddEvenSort.dir/flags.make
CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o: oddEvenSort.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/scratch/user/erichaddad/Cuda Implementation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o"
	/sw/eb/sw/CUDA/9.2.88/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c "/scratch/user/erichaddad/Cuda Implementation/oddEvenSort.cu" -o CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o

CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target oddEvenSort
oddEvenSort_OBJECTS = \
"CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o"

# External object files for target oddEvenSort
oddEvenSort_EXTERNAL_OBJECTS =

CMakeFiles/oddEvenSort.dir/cmake_device_link.o: CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: CMakeFiles/oddEvenSort.dir/build.make
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: /lib64/librt.so
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: /lib64/libpthread.so
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: /lib64/libdl.so
CMakeFiles/oddEvenSort.dir/cmake_device_link.o: CMakeFiles/oddEvenSort.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/scratch/user/erichaddad/Cuda Implementation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/oddEvenSort.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/oddEvenSort.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/oddEvenSort.dir/build: CMakeFiles/oddEvenSort.dir/cmake_device_link.o

.PHONY : CMakeFiles/oddEvenSort.dir/build

# Object files for target oddEvenSort
oddEvenSort_OBJECTS = \
"CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o"

# External object files for target oddEvenSort
oddEvenSort_EXTERNAL_OBJECTS =

oddEvenSort: CMakeFiles/oddEvenSort.dir/oddEvenSort.cu.o
oddEvenSort: CMakeFiles/oddEvenSort.dir/build.make
oddEvenSort: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
oddEvenSort: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
oddEvenSort: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
oddEvenSort: /lib64/librt.so
oddEvenSort: /lib64/libpthread.so
oddEvenSort: /lib64/libdl.so
oddEvenSort: CMakeFiles/oddEvenSort.dir/cmake_device_link.o
oddEvenSort: CMakeFiles/oddEvenSort.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/scratch/user/erichaddad/Cuda Implementation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable oddEvenSort"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/oddEvenSort.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/oddEvenSort.dir/build: oddEvenSort

.PHONY : CMakeFiles/oddEvenSort.dir/build

CMakeFiles/oddEvenSort.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/oddEvenSort.dir/cmake_clean.cmake
.PHONY : CMakeFiles/oddEvenSort.dir/clean

CMakeFiles/oddEvenSort.dir/depend:
	cd "/scratch/user/erichaddad/Cuda Implementation" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/scratch/user/erichaddad/Cuda Implementation" "/scratch/user/erichaddad/Cuda Implementation" "/scratch/user/erichaddad/Cuda Implementation" "/scratch/user/erichaddad/Cuda Implementation" "/scratch/user/erichaddad/Cuda Implementation/CMakeFiles/oddEvenSort.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/oddEvenSort.dir/depend

