.PHONY: all clean
all: build/gpt2

build/gpt2: src/main.cu src/gpt2.cu
	nvcc src/main.cu src/gpt2.cu -o build/gpt2 -I./third_party -std=c++17

clean:
	rm -f build/gpt2