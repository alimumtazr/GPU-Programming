# Compiler
CXX = g++
# Compiler flags: optimize, show all warnings, C++17 standard
CXXFLAGS = -O2 -Wall -std=c++17

# Target executable
TARGET = matrix_add_cpu

# Source files
SRC = matrix_add_cpu.cpp

# Default target: build executable
all: $(TARGET)

# Rule to compile the program
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean build artifacts
clean:
	rm -f $(TARGET)

