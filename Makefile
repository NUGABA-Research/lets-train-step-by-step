build/main : src/weight_loader.cu third_party/cJSON/cJSON.c
	nvcc src/weight_loader.cu third_party/cJSON/cJSON.c -o build/main

.PHONY: all clean

all: build/main

clean:
	rm -f build/main