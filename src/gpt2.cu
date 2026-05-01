#include <cuda_runtime.h>

#include <cstdint>

#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "gpt2.h"

using json = nlohmann::json;

GPT2Config parse_config(const std::string& config_path);
std::pair<json, uint64_t> parse_safetensor_header(std::ifstream& weights_file);
void *load_weights_to_gpu(std::ifstream& weights_file, size_t weights_total_size);
GPT2Weight parse_safetensor_tensor(const json& parsed_header, void* params_memory, const GPT2Config& config);

GPT2::GPT2(const std::string& weights_path, const std::string& config_path) {
    config = parse_config(config_path);

    std::ifstream weights_file(weights_path, std::ios::binary);
    if (weights_file.is_open() == false) {
        throw std::runtime_error("Model checkpoint does not exist.");
    }

    auto [parsed_header, header_size] = parse_safetensor_header(weights_file);
    size_t weights_total_size = std::filesystem::file_size(weights_path) - 8 - header_size;
    std::cout << "Size of weights: " << weights_total_size << '\n';

    params_memory = load_weights_to_gpu(weights_file, weights_total_size);

    weight = parse_safetensor_tensor(parsed_header, params_memory, config);
}

GPT2::~GPT2() {
    cudaFree(params_memory);
    free(weight.blocks);
}

GPT2Config parse_config(const std::string& config_path) {
    GPT2Config config;

    std::ifstream config_file(config_path);
    json config_json = json::parse(config_file);

    config.n_layer = config_json["n_layer"];
    config.n_head = config_json["n_head"];
    config.d_model = config_json["n_embd"];
    config.vocab_size = config_json["vocab_size"];
    config.seq_len = config_json["n_ctx"];

    return config;
}

std::pair<json, uint64_t> parse_safetensor_header(std::ifstream& weights_file) {
    uint64_t header_size = 0;
    weights_file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
    
    std::cout << "Header size: " << header_size << '\n';

    std::string header(header_size, 0);
    weights_file.read(header.data(), header_size);

    json parsed_header = json::parse(header);

    std::cout << "Header content: " << parsed_header.dump() << '\n';

    return std::make_pair(parsed_header, header_size);
}

void *load_weights_to_gpu(std::ifstream& weights_file, size_t weights_total_size) {
    void *host_buffer_address = nullptr;
    void *gpu_buffer_address = nullptr;
    
    cudaMallocHost(&host_buffer_address, weights_total_size);
    weights_file.read(reinterpret_cast<char*>(host_buffer_address), weights_total_size);
    
    cudaMalloc(&gpu_buffer_address, weights_total_size);
    cudaMemcpy(gpu_buffer_address, host_buffer_address, weights_total_size, cudaMemcpyHostToDevice);

    cudaFreeHost(host_buffer_address);

    return gpu_buffer_address;
}

GPT2Weight parse_safetensor_tensor(const json& parsed_header, void* params_memory, const GPT2Config& config) {
    GPT2Weight weight;

    weight.blocks = (TransformerBlockWeight*)calloc(config.n_layer, sizeof(TransformerBlockWeight));
    
    for (auto& [name, item] : parsed_header.items()) {
        if (name == "__metadata__") continue;

        size_t offset = item["data_offsets"][0].get<size_t>();
        float* target = static_cast<float *>(static_cast<char *>(params_memory) + offset);

        if (name == "wte.weight") {
            weight.wte_weight = target;
        }
        else if (name == "wpe.weight") {
            weight.wpe_weight = target;
        }
        else if (name == "ln_f.weight") {
            weight.ln_f_weight = target;
        }
        else if (name == "ln_f.bias") {
            weight.ln_f_bias = target;
        }
        else if (name.substr(0, 2) == "h.") {
            std::istringstream ss(name);

            std::string layer_depth_str;
            int layer_depth = 0;

            std::string layer_type;

            std::getline(ss, layer_depth_str, '.');
            std::getline(ss, layer_depth_str, '.');

            layer_depth = std::stoi(layer_depth_str);

            if (layer_depth >= config.n_layer) {
                throw std::runtime_error("Layer depth out of range: " + name);
            }
            
            std::getline(ss, layer_type);

            if (layer_type == "ln_1.weight") {
                weight.blocks[layer_depth].ln_1_weight = target;
            }
            else if (layer_type == "ln_1.bias") {
                weight.blocks[layer_depth].ln_1_bias = target;
            }
            else if (layer_type == "attn.c_attn.weight") {
                weight.blocks[layer_depth].attn_c_attn_weight = target;
            }
            else if (layer_type == "attn.c_attn.bias") {
                weight.blocks[layer_depth].attn_c_attn_bias = target;
            }
            else if (layer_type == "attn.c_proj.weight") {
                weight.blocks[layer_depth].attn_c_proj_weight = target;
            }
            else if (layer_type == "attn.c_proj.bias") {
                weight.blocks[layer_depth].attn_c_proj_bias = target;
            }
            else if (layer_type == "attn.bias") {
                weight.blocks[layer_depth].attn_bias = target;
            }
            else if (layer_type == "ln_2.weight") {
                weight.blocks[layer_depth].ln_2_weight = target;
            }
            else if (layer_type == "ln_2.bias") {
                weight.blocks[layer_depth].ln_2_bias = target;
            }
            else if (layer_type == "mlp.c_fc.weight") {
                weight.blocks[layer_depth].mlp_c_fc_weight = target;
            }
            else if (layer_type == "mlp.c_fc.bias") {
                weight.blocks[layer_depth].mlp_c_fc_bias = target;
            }
            else if (layer_type == "mlp.c_proj.weight") {
                weight.blocks[layer_depth].mlp_c_proj_weight = target;
            }
            else if (layer_type == "mlp.c_proj.bias") {
                weight.blocks[layer_depth].mlp_c_proj_bias = target;
            }
            else {
                throw std::runtime_error("Unknown tensor: " + name);
            }
        }
        else {
            throw std::runtime_error("Unknown tensor: " + name);
        }
    }

    return weight;
}