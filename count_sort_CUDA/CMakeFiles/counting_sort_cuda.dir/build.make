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
CMAKE_SOURCE_DIR = /scratch/user/samuli.hirvilampi/project/CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/user/samuli.hirvilampi/project/CUDA

# Include any dependencies generated for this target.
include CMakeFiles/counting_sort_cuda.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/counting_sort_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/counting_sort_cuda.dir/flags.make

CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o: CMakeFiles/counting_sort_cuda.dir/flags.make
CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o: counting_sort_cuda.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/user/samuli.hirvilampi/project/CUDA/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o"
	/sw/eb/sw/CUDA/9.2.88/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /scratch/user/samuli.hirvilampi/project/CUDA/counting_sort_cuda.cu -o CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o

CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target counting_sort_cuda
counting_sort_cuda_OBJECTS = \
"CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o"

# External object files for target counting_sort_cuda
counting_sort_cuda_EXTERNAL_OBJECTS =

CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: CMakeFiles/counting_sort_cuda.dir/build.make
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: /lib64/librt.so
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: /lib64/libpthread.so
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: /lib64/libdl.so
CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o: CMakeFiles/counting_sort_cuda.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/user/samuli.hirvilampi/project/CUDA/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/counting_sort_cuda.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/counting_sort_cuda.dir/build: CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o

.PHONY : CMakeFiles/counting_sort_cuda.dir/build

# Object files for target counting_sort_cuda
counting_sort_cuda_OBJECTS = \
"CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o"

# External object files for target counting_sort_cuda
counting_sort_cuda_EXTERNAL_OBJECTS =

counting_sort_cuda: CMakeFiles/counting_sort_cuda.dir/counting_sort_cuda.cu.o
counting_sort_cuda: CMakeFiles/counting_sort_cuda.dir/build.make
counting_sort_cuda: /scratch/group/csce435-f23/Caliper/caliper/lib64/libcaliper.so.2.11.0-dev
counting_sort_cuda: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/libmpicxx.so
counting_sort_cuda: /sw/eb/sw/impi/2019.9.304-iccifort-2020.4.304/intel64/lib/release/libmpi.so
counting_sort_cuda: /lib64/librt.so
counting_sort_cuda: /lib64/libpthread.so
counting_sort_cuda: /lib64/libdl.so
counting_sort_cuda: CMakeFiles/counting_sort_cuda.dir/cmake_device_link.o
counting_sort_cuda: CMakeFiles/counting_sort_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/user/samuli.hirvilampi/project/CUDA/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable counting_sort_cuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/counting_sort_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/counting_sort_cuda.dir/build: counting_sort_cuda

.PHONY : CMakeFiles/counting_sort_cuda.dir/build

CMakeFiles/counting_sort_cuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/counting_sort_cuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/counting_sort_cuda.dir/clean

CMakeFiles/counting_sort_cuda.dir/depend:
	cd /scratch/user/samuli.hirvilampi/project/CUDA && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/user/samuli.hirvilampi/project/CUDA /scratch/user/samuli.hirvilampi/project/CUDA /scratch/user/samuli.hirvilampi/project/CUDA /scratch/user/samuli.hirvilampi/project/CUDA /scratch/user/samuli.hirvilampi/project/CUDA/CMakeFiles/counting_sort_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/counting_sort_cuda.dir/depend

