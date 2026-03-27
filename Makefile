build/weight_loader : src/weight_loader.cu third_party/cJSON/cJSON.c
	nvcc src/weight_loader.cu third_party/cJSON/cJSON.c -o build/weight_loader

.PHONY: all clean

all: build/weight_loader

clean:
	rm -f build/weight_loader