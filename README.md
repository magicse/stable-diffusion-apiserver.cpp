# stable-diffusion-apiserver.cpp Extended API
This project is an experimental C/C++ API server for Stable Diffusion using stable-diffusion.cpp as the backend.
This is a work-in-progress C/C++ API server for [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp).  
The goal is to allow dynamic loading/unloading of components (VAE, LoRA, ControlNet, etc.) and enable runtime configuration without needing to reload the entire model.

The main goal is to provide a more flexible runtime where you don't have to reload the entire model just to adjust some settings or switch components (like VAE, LoRA, ControlNet, etc.).

## Key Features (in progress):
- lightweight HTTP API wrapper using [`cpp-httplib`](https://github.com/yhirose/cpp-httplib)
- Load/unload individual model components dynamically
- Adjust generation parameters without restarting the whole pipeline
- Expose core generation methods (txt2img, img2img, etc.) via extended context
- Optional component placement (e.g., keep CLIP/ControlNet/VAEs on CPU)


## Current Status
This is still a work in progress. Expect unstable behavior and incomplete features. Any feedback or contributions are welcome.

## Build Instructions
Project Structure
```
sd_server/
├── CMakeLists.txt                # Build configuration
├── sd_server.cpp                 # Main entry point (server)
├── stable-diffusion.cpp/         # Submodule (core + extended API)
│   ├── stable_diffusion_extended.h
│   └── stable_diffusion_extended.cpp
├── httplib/                      # Header-only HTTP library
```

### Linux
```bash
git clone --recurse-submodules https://github.com/magicse/stable-diffusion-apiserver.cpp.git
cd sd_server
mkdir build && cd build
cmake ..
make
./sd_server
```
### Windows (MinGW)
```
git clone --recurse-submodules https://github.com/magicse/stable-diffusion-apiserver.cpp.git
cd sd_server
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
sd_server.exe
```

