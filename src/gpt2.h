struct GPT2Config {
    int n_layer;   // 12
    int n_head;    // 12
    int d_model;   // 768
    int vocab_size; // 50257
    int seq_len;   // 1024
};

struct TransformerBlockWeight {
    float *h_n_ln_1_weight;
    float *h_n_ln_1_bias;

    float *h_n_attn_c_attn_weight;
    float *h_n_attn_c_attn_bias;
    float *h_n_attn_c_proj_weight;
    float *h_n_attn_c_proj_bias;

    float *h_n_ln_2_weight;
    float *h_n_ln_2_bias;

    float *h_n_mlp_c_fc_weight;
    float *h_n_mlp_c_fc_bias;
    float *h_n_mlp_c_proj_weight;
    float *h_n_mlp_c_proj_bias;
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
};