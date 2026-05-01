#include <cstdio>
#include <cstring>
#include <cstdint>

#include <string>

#include <cuda_runtime.h>

#include "../third_party/cJSON/cJSON.h"

#include "gpt2.h"

int parse_safetensor_header(FILE *fp, void *host_buffer_for_weights, long &weights_total_size);

int main(void) {
    FILE *fp = NULL;
    void *host_buffer_for_weights = NULL;
    long weights_total_size = 0;

    if (parse_safetensor_header(fp, host_buffer_for_weights, weights_total_size) == -1) {
        return -1;
    }

    GPT2 gpt2("../weights/model.safetensors", "../weights/config.json");

    cudaMalloc(&gpt2->params_memory, weights_total_size);
    cudaMemcpy(gpt2->params_memory, host_buffer_for_weights, weights_total_size, cudaMemcpyHostToDevice);

    // We will not do the sanity check here.
    cJSON *tensor = parsed_header->child;
    while (tensor != NULL) {
        const char *tensor_name = tensor->string;
        
        if (strcmp(tensor_name, "__metadata__") == 0) {
            tensor = tensor->next;
            continue;
        }

        cJSON *offsets = cJSON_GetObjectItem(tensor, "data_offsets");
        if (offsets == NULL) {
            fprintf(stderr, "Failed to get object item \"data_offsets\".");
            return -1;
        }

        cJSON *offset_start_ptr = cJSON_GetArrayItem(offsets, 0);
        if (offset_start_ptr == NULL) {
            fprintf(stderr, "Failed to get 0th item of the \"data_offsets\" array.");
            return -1;
        }
        long offset_start_value = offset_start_ptr->valueint;

        float* target_address = (float*)((char*)gpt2->params_memory + offset_start_value);
        
        if (strcmp(tensor_name, "wte.weight") == 0) {
            gpt2->weight.wte_weight = target_address;
        }
        else if (strcmp(tensor_name, "wpe.weight") == 0) {
            gpt2->weight.wpe_weight = target_address;
        } 
        else if (strcmp(tensor_name, "ln_f.weight") == 0) {
            gpt2->weight.ln_f_weight = target_address;
        } 
        else if (strcmp(tensor_name, "ln_f.bias") == 0) {
            gpt2->weight.ln_f_bias = target_address;
        } 
        else if (strncmp(tensor_name, "h.", 2) == 0) {
            unsigned int layer_depth;
            char layer_type[19];

            if (sscanf(tensor_name, "h.%u.%s", &layer_depth, layer_type) != 2) {
                fprintf(stderr, "Parsing a tensor name in a transformer block failed.");
                return -1;
            }

            if (layer_depth >= gpt2->config.n_layer) {
                fprintf(stderr, "The depth of the layer is out of range.");
                return -1;
            }

            if (strcmp(layer_type, "ln_1.weight") == 0) {
                gpt2->weight.blocks[layer_depth].ln_1_weight = target_address;
            }
            else if (strcmp(layer_type, "ln_1.bias") == 0) {
                gpt2->weight.blocks[layer_depth].ln_1_bias = target_address;
            }
            else if (strcmp(layer_type, "attn.c_attn.weight") == 0) {
                gpt2->weight.blocks[layer_depth].attn_c_attn_weight = target_address;
            }
            else if (strcmp(layer_type, "attn.c_attn.bias") == 0) {
                gpt2->weight.blocks[layer_depth].attn_c_attn_bias = target_address;
            }
            else if (strcmp(layer_type, "attn.c_proj.weight") == 0) {
                gpt2->weight.blocks[layer_depth].attn_c_proj_weight = target_address;
            }
            else if (strcmp(layer_type, "attn.c_proj.bias") == 0) {
                gpt2->weight.blocks[layer_depth].attn_c_proj_bias = target_address;
            }
            else if (strcmp(layer_type, "attn.bias") == 0) {
                gpt2->weight.blocks[layer_depth].attn_bias = target_address;
            }
            else if (strcmp(layer_type, "ln_2.weight") == 0) {
                gpt2->weight.blocks[layer_depth].ln_2_weight = target_address;
            }
            else if (strcmp(layer_type, "ln_2.bias") == 0) {
                gpt2->weight.blocks[layer_depth].ln_2_bias = target_address;
            }
            else if (strcmp(layer_type, "mlp.c_fc.weight") == 0) {
                gpt2->weight.blocks[layer_depth].mlp_c_fc_weight = target_address;
            }
            else if (strcmp(layer_type, "mlp.c_fc.bias") == 0) {
                gpt2->weight.blocks[layer_depth].mlp_c_fc_bias = target_address;
            }
            else if (strcmp(layer_type, "mlp.c_proj.weight") == 0) {
                gpt2->weight.blocks[layer_depth].mlp_c_proj_weight = target_address;
            }
            else if (strcmp(layer_type, "mlp.c_proj.bias") == 0) {
                gpt2->weight.blocks[layer_depth].mlp_c_proj_bias = target_address;
            }
            else {
                fprintf(stderr, "An unknown type tensor found.");
                return -1;
            }
        }
        else {
            fprintf(stderr, "An unknown type tensor found.");
            return -1;
        }
        
        tensor = tensor->next;
    }

    cudaFreeHost(host_buffer_for_weights);
    free(header_content);

    cJSON_Delete(parsed_header);
    fclose(fp);

    return 0;
}

int parse_safetensor_header(FILE *fp, void *host_buffer_for_weights, long &weights_total_size) {
    fp = fopen("weights/model.safetensors", "rb");
    
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

    weights_total_size = end_pointer - weights_start_pointer;
    printf("Size of weights: %ld\n", weights_total_size);

    cudaMallocHost(&host_buffer_for_weights, weights_total_size);

    if (fseek(fp, weights_start_pointer, SEEK_SET) != 0) {
        fprintf(stderr, "Seeking weights start pointer of the file failed.");
        return -1;
    }

    fread(host_buffer_for_weights, 1, weights_total_size, fp);
}