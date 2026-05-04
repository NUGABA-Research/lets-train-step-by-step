#include <string>

struct GPT2Config {
    int n_layer;   // 12
    int n_head;    // 12
    int d_model;   // 768
    int vocab_size; // 50257
    int seq_len;   // 1024
};

struct TransformerBlockWeight {
    float *ln_1_weight;
    float *ln_1_bias;

    float *attn_c_attn_weight;
    float *attn_c_attn_bias;
    float *attn_c_proj_weight;
    float *attn_c_proj_bias;
    float *attn_bias;

    float *ln_2_weight;
    float *ln_2_bias;

    float *mlp_c_fc_weight;
    float *mlp_c_fc_bias;
    float *mlp_c_proj_weight;
    float *mlp_c_proj_bias;
};

struct GPT2Weight {
    float *wte_weight;
    float *wpe_weight;

    TransformerBlockWeight* blocks;

    float *ln_f_weight;
    float *ln_f_bias;
};

struct GPT2 {
    GPT2Config config;
    GPT2Weight weight;

    void *params_memory;

    GPT2(const std::string& weights_path, const std::string& config_path);
    ~GPT2();
};