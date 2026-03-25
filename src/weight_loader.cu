#include <cstdio>
#include <cstring>
#include <cstdint>

#include <cuda_runtime.h>

#include "../third_party/cJSON/cJSON.h"

// __global__ void hello() {
//     printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
// }

int main(void) {
    FILE *fp = fopen("weights/model.safetensors", "rb");
    
    if (fp == NULL) {
        perror("Model checkpoint does not exist.");
        return -1;
    }

    char header_size_buffer[8];
    uint64_t header_size = 0;

    fread(header_size_buffer, 1, 8, fp); // read and save to 'header_size_buffer', '1' byte-sized '8' elements, from 'fp'
    memcpy(&header_size, header_size_buffer, sizeof(uint64_t));

    //cJSON *json = cJSON_Parse(header);
    
    printf("%lu\n", header_size);

    // hello<<<1, 4>>>();
    // cudaError_t err = cudaDeviceSynchronize();

    // if (err != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(err));
    //     return 1;
    // }

    return 0;
}