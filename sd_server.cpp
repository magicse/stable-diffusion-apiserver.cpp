#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include <filesystem>
#include <atomic>
#include <csignal>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// httplib.h - include path
#include "httplib.h"

// headers stable-diffusion.cpp 
#include "stable-diffusion.h"
#include "stable-diffusion-extended.h"

#define STR2(x) #x
#define STR(x) STR2(x)

#pragma message("_WIN32_WINNT=" STR(_WIN32_WINNT)) 

class StableDiffusionServer {
private:
    sd_ctx_t* sd_ctx;
    std::mutex generation_mutex;
    bool model_loaded = false;
    std::string model_path;
    
public:
    StableDiffusionServer() : sd_ctx(nullptr) {}
    
    ~StableDiffusionServer() {
        cleanup();
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(generation_mutex);
        if (sd_ctx) {
            free_sd_ctx(sd_ctx);
            sd_ctx = nullptr;
        }
        model_loaded = false;
    }

    bool is_model_loaded() const {
        return model_loaded;
    }
    
    bool load_model(const std::string& path) {
        std::lock_guard<std::mutex> lock(generation_mutex);
        
        // Free old context
        if (sd_ctx) {
            free_sd_ctx(sd_ctx);
            sd_ctx = nullptr;
        }
        
        // Model parmaters
        sd_ctx = new_sd_ctx(path.c_str(),     // model_path
                            "",               // clip_l_path
                            "",               // clip_g_path
                            "",               // t5xxl_path
                            "",               // diffusion_model_path
                            "",               // vae_path
                            "",               // taesd_path
                            "",               // control_net_path
                            "",               // lora_model_dir
                            "",               // embed_dir
                            "",               // stacked_id_embed_dir
                            false,            // vae_decode_only
                            true,             // vae_tiling
                            false,             // free_params_immediately unload weights after generation
                            6,                // n_threads
                            SD_TYPE_F16,      // wtype
                            STD_DEFAULT_RNG,  // RNG без CUDA
                            KARRAS,           // schedule
                            false,            // keep_clip_on_cpu
                            false,            // keep_control_net_cpu
                            false,            // keep_vae_on_cpu
                            false             // diffusion_flash_attn
        );
        
        model_loaded = (sd_ctx != nullptr);
        if (model_loaded) {
            model_path = path;
            std::cout << "Model loaded successfully: " << path << std::endl;
        } else {
            std::cout << "Failed to load model: " << path << std::endl;
        }
        
        return model_loaded;
    }


std::vector<std::string> generate_image(const std::string& prompt,
                                        const std::string& negative_prompt = "",
                                        int width = 512,
                                        int height = 512,
                                        int steps = 20,
                                        float cfg_scale = 7.0f,
                                        int seed = -1,
                                        int batch_count = 1) {
    std::lock_guard<std::mutex> lock(generation_mutex);
    std::vector<std::string> filenames;

    if (!model_loaded || !sd_ctx) {
        std::cout << "Model not loaded or context is null" << std::endl;
        return filenames;
    }

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::cout << "Starting generation with prompt: " << prompt << std::endl;

    sd_image_t* results = nullptr;

    try {
        results = txt2img(sd_ctx,
                          prompt.c_str(),
                          negative_prompt.c_str(),
                          -1,
                          cfg_scale,
                          1.0f,
                          0.0f,
                          width,
                          height,
                          EULER_A,
                          steps,
                          static_cast<int64_t>(seed),
                          batch_count,
                          nullptr,
                          0.0f,
                          0.0f,
                          false,
                          "",
                          nullptr,
                          0,
                          0.0f,
                          0.0f,
                          1.0f);

        if (!results) {
            std::cout << "txt2img returned null" << std::endl;
            return filenames;
        }

        for (int i = 0; i < batch_count; ++i) {
            if (!results[i].data) {
                std::cout << "Image " << i << " is null, skipping" << std::endl;
                continue;
            }

            std::string filename = "generated_" + std::to_string(ms + i) + ".png";
            int result = stbi_write_png(filename.c_str(),
                                        results[i].width,
                                        results[i].height,
                                        results[i].channel,
                                        results[i].data,
                                        results[i].width * results[i].channel);

            if (result) {
                filenames.push_back(filename);
                std::cout << "Saved: " << filename << std::endl;
            } else {
                std::cout << "Failed to save image: " << i << std::endl;
            }
        }

        // Memory free
        for (int i = 0; i < batch_count; ++i) {
            if (results[i].data) {
                free(results[i].data);
                results[i].data = nullptr;
            }
        }
        free(results);
        results = nullptr;

    } catch (const std::exception& e) {
        std::cout << "Exception during generation: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception during generation" << std::endl;
    }

    std::cout << "Generation completed" << std::endl;
    return filenames;
}

    
    std::string generate_image_old(const std::string& prompt, 
                              const std::string& negative_prompt = "",
                              int width = 512, 
                              int height = 512,
                              int steps = 20,
                              float cfg_scale = 7.0f,
                              int seed = -1,
							  int batch_count = 1) {
        
        std::lock_guard<std::mutex> lock(generation_mutex);
        
        if (!model_loaded || !sd_ctx) {
            std::cout << "Model not loaded or context is null" << std::endl;
            return "";
        }
        
        // File name generation
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::string filename = "generated_" + std::to_string(ms) + ".png";

        std::cout << "Starting generation with prompt: " << prompt << std::endl;
        
        // init ptr
        sd_image_t* results = nullptr;
        
        try {
            // Параметры генерации
            results = txt2img(sd_ctx,
                prompt.c_str(),
                negative_prompt.c_str(),
                -1,          // clip_skip
                cfg_scale,   // cfg
                1.0f,        // guidance
                0.0f,        // eta
                width,
                height,
                EULER_A,     // sample_method
                steps,
                static_cast<int64_t>(seed),
                batch_count, // batch_count
                nullptr,     // control_cond
                0.0f,        // control_strength
                0.0f,        // style_strength
                false,       // normalize_input
                "",          // input_id_images_path
                nullptr,     // skip_layers
                0,           // skip_layers_count
                0.0f,        // slg_scale
                0.0f,        // skip_layer_start
                1.0f         // skip_layer_end
            );
            
            std::cout << "txt2img returned, checking results..." << std::endl;
            
            if (results && results->data) {
                std::cout << "Image data received: " << results->width << "x" << results->height 
                         << " channels: " << results->channel << std::endl;



                // Check data
                if (results->width > 0 && results->height > 0 && results->channel > 0) {
                    // Save Image
                    int result = stbi_write_png(filename.c_str(), 
                                              results->width, 
                                              results->height, 
                                              results->channel, 
                                              results->data, 
                                              results->width * results->channel);
                    
                    if (result) {
                        std::cout << "Image saved successfully: " << filename << std::endl;
                    } else {
                        std::cout << "Failed to save image: " << filename << std::endl;
                        filename = "";
                    }
                } else {
                    std::cout << "Invalid image dimensions received" << std::endl;
                    filename = "";
                }
				
                
                if (results->data) {
                    free(results->data);
                    results->data = nullptr;
                }
                free(results);
                results = nullptr;
                
            } else {
                std::cout << "txt2img returned null or no data" << std::endl;
                filename = "";
                
                // if results not null, but data null - free
                if (results) {
                    free(results);
                    results = nullptr;
                }
            }
            
        } catch (const std::exception& e) {
            std::cout << "Exception during generation: " << e.what() << std::endl;
            filename = "";
            
            // Exeption free memoty
            if (results) {
                if (results->data) {
                    free(results->data);
                }
                free(results);
            }
        } catch (...) {
            std::cout << "Unknown exception during generation" << std::endl;
            filename = "";
            
            // Unknown exeption free memory
            if (results) {
                if (results->data) {
                    free(results->data);
                }
                free(results);
            }
        }
        
        std::cout << "Generation completed" << std::endl;
        return filename;
    }
};
