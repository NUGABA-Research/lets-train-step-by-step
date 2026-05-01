#include "src/gpt2.h"

int main(void) {
    GPT2 gpt2("../weights/model.safetensors", "../weights/config.json");
    
    return 0;
}