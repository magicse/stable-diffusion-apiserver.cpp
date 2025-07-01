#ifndef __STABLE_DIFFUSION_EXTENDED_H__
#define __STABLE_DIFFUSION_EXTENDED_H__

#include "stable-diffusion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enumeration of component types
typedef enum {
    SD_COMPONENT_FULL_MODEL,
    SD_COMPONENT_CLIP_L,
    SD_COMPONENT_CLIP_G,
    SD_COMPONENT_T5XXL,
    SD_COMPONENT_DIFFUSION_MODEL,
    SD_COMPONENT_VAE,
    SD_COMPONENT_TAESD,
    SD_COMPONENT_CONTROL_NET,
    SD_COMPONENT_LORA_MODEL,
    SD_COMPONENT_EMBED_DIR,
    SD_COMPONENT_ID_EMBED_DIR,
    SD_COMPONENT_COUNT
} sd_component_type_t;

// Structure for an individual component
typedef struct {
    void* model_data;
    char* path;
    bool is_loaded;
    enum sd_type_t wtype;  // quantization type for this component
} sd_component_t;

// Structure for the extended context
typedef struct {
    // Model components
    sd_component_t components[SD_COMPONENT_COUNT];
    
    // Context parameters (copied from your new_sd_ctx)
    //char* embed_dir;
    //char* stacked_id_embed_dir;
    bool vae_decode_only;
    bool vae_tiling;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;  // global default quantization type
    enum rng_type_t rng_type;
    enum schedule_t schedule;
    bool keep_clip_on_cpu;
    bool keep_control_net_cpu;
    bool keep_vae_on_cpu;
    bool diffusion_flash_attn;
    bool chroma_use_dit_mask;
    bool chroma_use_t5_mask;
    int chroma_t5_mask_pad;
    
    // Internal SD context
    sd_ctx_t* base_ctx;
    bool is_initialized;
} sd_extended_ctx_t;

// =================
// CORE FUNCTIONS
// =================

// Create and free extended context
SD_API sd_extended_ctx_t* new_sd_extended_ctx(void);
SD_API void free_sd_extended_ctx(sd_extended_ctx_t* ctx);

// =================
// COMPONENT LOADING
// =================

// Load individual components
SD_API bool sd_load_clip_l(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_clip_g(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_t5xxl(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_diffusion_model(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_vae(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_taesd(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_control_net(sd_extended_ctx_t* ctx, const char* path, enum sd_type_t wtype);
SD_API bool sd_load_lora_model(sd_extended_ctx_t* ctx, const char* path);

// Simplified loading functions (use global wtype)
SD_API bool sd_load_clip_l_simple(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_load_clip_g_simple(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_load_t5xxl_simple(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_load_diffusion_model_simple(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_load_vae_simple(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_load_taesd_simple(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_load_control_net_simple(sd_extended_ctx_t* ctx, const char* path);

// =================
// COMPONENT UNLOADING
// =================

SD_API void sd_unload_clip_l(sd_extended_ctx_t* ctx);
SD_API void sd_unload_clip_g(sd_extended_ctx_t* ctx);
SD_API void sd_unload_t5xxl(sd_extended_ctx_t* ctx);
SD_API void sd_unload_diffusion_model(sd_extended_ctx_t* ctx);
SD_API void sd_unload_vae(sd_extended_ctx_t* ctx);
SD_API void sd_unload_taesd(sd_extended_ctx_t* ctx);
SD_API void sd_unload_control_net(sd_extended_ctx_t* ctx);
SD_API void sd_unload_lora_model(sd_extended_ctx_t* ctx);
SD_API void sd_unload_component(sd_extended_ctx_t* ctx, sd_component_type_t type);

// =================
// PARAMETER CONTROL
// =================

// VAE settings
SD_API bool sd_set_vae_decode_only(sd_extended_ctx_t* ctx, bool value);
SD_API bool sd_set_vae_tiling(sd_extended_ctx_t* ctx, bool value);

// Performance settings
SD_API bool sd_set_threads(sd_extended_ctx_t* ctx, int n_threads);
SD_API bool sd_set_free_params_immediately(sd_extended_ctx_t* ctx, bool value);

// Data type settings
SD_API bool sd_set_wtype(sd_extended_ctx_t* ctx, enum sd_type_t wtype);
SD_API bool sd_set_rng_type(sd_extended_ctx_t* ctx, enum rng_type_t rng_type);
SD_API bool sd_set_schedule(sd_extended_ctx_t* ctx, enum schedule_t schedule);

// Memory placement settings
SD_API bool sd_set_keep_clip_on_cpu(sd_extended_ctx_t* ctx, bool value);
SD_API bool sd_set_keep_control_net_cpu(sd_extended_ctx_t* ctx, bool value);
SD_API bool sd_set_keep_vae_on_cpu(sd_extended_ctx_t* ctx, bool value);

// Optimization settings
SD_API bool sd_set_diffusion_flash_attn(sd_extended_ctx_t* ctx, bool value);

// Chroma settings
SD_API bool sd_set_chroma_use_dit_mask(sd_extended_ctx_t* ctx, bool value);
SD_API bool sd_set_chroma_use_t5_mask(sd_extended_ctx_t* ctx, bool value);
SD_API bool sd_set_chroma_t5_mask_pad(sd_extended_ctx_t* ctx, int pad);

// Directory settings
SD_API bool sd_set_embed_dir(sd_extended_ctx_t* ctx, const char* path);
SD_API bool sd_set_stacked_id_embed_dir(sd_extended_ctx_t* ctx, const char* path);

// =================
// INFORMATIONAL FUNCTIONS
// =================

// Check component state
SD_API bool sd_is_component_loaded(sd_extended_ctx_t* ctx, sd_component_type_t type);
SD_API const char* sd_get_component_path(sd_extended_ctx_t* ctx, sd_component_type_t type);
SD_API enum sd_type_t sd_get_component_wtype(sd_extended_ctx_t* ctx, sd_component_type_t type);

// Output information
SD_API void sd_list_loaded_components(sd_extended_ctx_t* ctx);
SD_API void sd_print_component_info(sd_extended_ctx_t* ctx, sd_component_type_t type);

// Context readiness check
SD_API bool sd_is_ready_for_generation(sd_extended_ctx_t* ctx);
SD_API const char* sd_get_missing_components(sd_extended_ctx_t* ctx);

// =================
// CORE OPERATIONS
// =================

// Rebuild context (apply all changes)
SD_API bool sd_rebuild_context(sd_extended_ctx_t* ctx);

// Get internal context (for compatibility)
SD_API sd_ctx_t* sd_get_base_context(sd_extended_ctx_t* ctx);

// =================
// GENERATION FUNCTIONS
// =================

// Extended text-to-image function (full parameter control)
SD_API sd_image_t* sd_txt2img_extended(sd_extended_ctx_t* ctx,
                                      const char* prompt,
                                      const char* negative_prompt,
                                      int clip_skip,
                                      float cfg_scale,
                                      float guidance,
                                      float eta,
                                      int width,
                                      int height,
                                      enum sample_method_t sample_method,
                                      int sample_steps,
                                      int64_t seed,
                                      int batch_count,
                                      const sd_image_t* control_cond,
                                      float control_strength,
                                      float style_strength,
                                      bool normalize_input,
                                      const char* input_id_images_path,
                                      int* skip_layers,
                                      size_t skip_layers_count,
                                      float slg_scale,
                                      float skip_layer_start,
                                      float skip_layer_end);

SD_API sd_image_t* sd_img2img_extended(sd_extended_ctx_t* ctx,
                                      sd_image_t init_image,
                                      sd_image_t mask_image,
                                      const char* prompt,
                                      const char* negative_prompt,
                                      int clip_skip,
                                      float cfg_scale,
                                      float guidance,
                                      float eta,
                                      int width,
                                      int height,
                                      enum sample_method_t sample_method,
                                      int sample_steps,
                                      float strength,
                                      int64_t seed,
                                      int batch_count,
                                      const sd_image_t* control_cond,
                                      float control_strength,
                                      float style_strength,
                                      bool normalize_input,
                                      const char* input_id_images_path,
                                      int* skip_layers,
                                      size_t skip_layers_count,
                                      float slg_scale,
                                      float skip_layer_start,
                                      float skip_layer_end);

SD_API sd_image_t* sd_img2vid_extended(sd_extended_ctx_t* ctx,
                                      sd_image_t init_image,
                                      int width,
                                      int height,
                                      int video_frames,
                                      int motion_bucket_id,
                                      int fps,
                                      float augmentation_level,
                                      float min_cfg,
                                      float cfg_scale,
                                      enum sample_method_t sample_method,
                                      int sample_steps,
                                      float strength,
                                      int64_t seed);

SD_API sd_image_t* sd_edit_extended(sd_extended_ctx_t* ctx,
                                   sd_image_t* ref_images,
                                   int ref_images_count,
                                   const char* prompt,
                                   const char* negative_prompt,
                                   int clip_skip,
                                   float cfg_scale,
                                   float guidance,
                                   float eta,
                                   int width,
                                   int height,
                                   enum sample_method_t sample_method,
                                   int sample_steps,
                                   float strength,
                                   int64_t seed,
                                   int batch_count,
                                   const sd_image_t* control_cond,
                                   float control_strength,
                                   float style_strength,
                                   bool normalize_input,
                                   int* skip_layers,
                                   size_t skip_layers_count,
                                   float slg_scale,
                                   float skip_layer_start,
                                   float skip_layer_end);

// =================
// UTILITY FUNCTIONS
// =================

// String to enum conversion
SD_API sd_component_type_t sd_component_type_from_string(const char* str);
SD_API const char* sd_component_type_to_string(sd_component_type_t type);

// Load full config from JSON/YAML file (optional)
SD_API bool sd_load_config_from_file(sd_extended_ctx_t* ctx, const char* config_path);
SD_API bool sd_save_config_to_file(sd_extended_ctx_t* ctx, const char* config_path);

#ifdef __cplusplus
}
#endif

#endif // __STABLE_DIFFUSION_EXTENDED_H__
