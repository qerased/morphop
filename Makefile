CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

BUILD_DIR := build
TARGET := $(BUILD_DIR)/morphlab
SRC := src/morphlab.cpp

.PHONY: all clean experiments gui

all: $(TARGET)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $< -o $@ $(OPENCV_LIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

experiments: $(TARGET)
	$(TARGET) --run-experiments --output-dir results

gui: $(TARGET)
	$(TARGET) --gui

clean:
	rm -rf $(BUILD_DIR)
