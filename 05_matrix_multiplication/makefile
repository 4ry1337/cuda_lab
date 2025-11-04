TARGET_EXEC ?= cuda_lab

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src

# CUDA settings
CUDA_PATH ?= /usr/local/cuda
NVCC ?= nvcc
CUDA_ARCH ?= sm_86

# Compiler flags
CFLAGS ?= -O0 -Wall
CXXFLAGS ?= -O0 -Wall 
NVCC_FLAGS := -arch=$(CUDA_ARCH) 
# -DVERIFY_RESULTS
# -DNAIVE
# -DGMEM
# -DSMEM
# -DDBLOCK
CUDA_OPT_FLAGS ?= -O3 

# Find all the C, C++, and CUDA files we want to compile
# Note the single quotes around the * expressions. The shell will incorrectly expand these otherwise, but we want to send the * directly to the find command.
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s' -or -name '*.cu')

# Prepends BUILD_DIR and appends .o to every src file
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
DEPS := $(OBJS:.o=.d)

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
CPPFLAGS := $(INC_FLAGS) -MMD -MP

# CUDA linking flags
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart

# The final build step.
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(CUDA_PATH)/include -c $< -o $@

# Build step for CUDA source
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_OPT_FLAGS) $(INC_FLAGS) -c $< -o $@

.PHONY: clean run all help debug

all: $(BUILD_DIR)/$(TARGET_EXEC)

clean:
	rm -rf $(BUILD_DIR)

run: $(BUILD_DIR)/$(TARGET_EXEC)
	./$(BUILD_DIR)/$(TARGET_EXEC)

debug:
	$(MAKE) CFLAGS="-O0 -g" CXXFLAGS="-O0 -g" CUDA_OPT_FLAGS="-O0 -g -G"

help:
	@echo "Available targets:"
	@echo "  all (default)  - Build the executable"
	@echo "  clean          - Remove build directory"
	@echo "  run            - Build and run the executable"
	@echo "  debug          - Build with debug symbols"
	@echo ""
	@echo "Customizable variables (use make VAR=value):"
	@echo "  CUDA_ARCH      - CUDA architecture (default: sm_86)"
	@echo "  CUDA_PATH      - CUDA toolkit path (default: /usr/local/cuda)"
	@echo "  TARGET_EXEC    - Output executable name (default: cuda_lab)"
	@echo "  BUILD_DIR      - Build directory (default: ./build)"
	@echo "  SRC_DIRS       - Source directory (default: ./src)"

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)
