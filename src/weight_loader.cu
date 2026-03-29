#include <cstdio>
#include <cstring>
#include <cstdint>

#include <string>

#include <cuda_runtime.h>

#include "../third_party/cJSON/cJSON.h"

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
    
    printf("Header size: %lu\n", header_size);

    std::string header(header_size, 0);
    fread(header.data(), 1, header_size, fp);

    cJSON *parsed_header = cJSON_Parse(header.c_str());
    if (parsed_header == NULL) {
        fprintf(stderr, "Safetensor information parsing failed.");
        return -1;
    }

    char* header_content = cJSON_Print(parsed_header);
    printf("Header content: %s\n", header_content);

    long weights_start_pointer = ftell(fp);
    if (weights_start_pointer == -1) {
        fprintf(stderr, "Reading weights start pointer of the file failed.");
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "Seeking the end pointer of the file failed.");
        return -1;
    }

    long end_pointer = ftell(fp);
    if (end_pointer == -1) {
        fprintf(stderr, "Reading end pointer of the file failed.");
        return -1;
    }

    long weights_total_size = end_pointer - weights_start_pointer;
    printf("Size of weights: %ld\n", weights_total_size);

    void *host_buffer_for_weights;

#ifdef CPU_ONLY
    host_buffer_for_weights = malloc(weights_total_size);
#else
    cudaMallocHost(&host_buffer_for_weights, weights_total_size);
#endif

    if (fseek(fp, weights_start_pointer, SEEK_SET) != 0) {
        fprintf(stderr, "Seeking weights start pointer of the file failed.");
        return -1;
    }

    fread(host_buffer_for_weights, 1, weights_total_size, fp);

    void* weights_pointer;

#ifdef CPU_ONLY
    printf("We can't load any weights to GPU, since this computer does not have one.\n");
#else
    cudaMalloc(&weights_pointer, weights_total_size);
    cudaMemcpy(weights_pointer, host_buffer_for_weights, weights_total_size, cudaMemcpyHostToDevice);
#endif

    // WIP

#ifdef CPU_ONLY
    free(host_buffer_for_weights);
#else
    cudaFree(weights_pointer);
    cudaFreeHost(host_buffer_for_weights);
#endif

    free(header_content);

    cJSON_Delete(parsed_header);
    fclose(fp);

    return 0;
}