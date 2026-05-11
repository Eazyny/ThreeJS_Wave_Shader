@'

# OceanShader WebGPU Engine

This folder is for the production WebGPU ocean engine.

The current `src/ocean/webgpu/probes/` files are working prototypes used to validate native WebGPU, Tessendorf spectrum synthesis, and GPU IFFT / Stockham FFT passes.

## Planned engine structure

- `shaders/`
  - WGSL shader modules for spectrum generation, FFT passes, displacement sampling, ocean surface rendering, and underwater rendering.

- `utils/`
  - Spectrum generation helpers, math utilities, bit-reversal helpers, and reusable WebGPU setup helpers.

- `pipelines/`
  - Functions for creating compute/render pipelines and bind groups.

- `buffers/`
  - Functions for creating uniform buffers, spectrum buffers, FFT buffers, sample buffers, and future texture outputs.

## Direction

The production path should gradually replace the giant prototype file with reusable engine modules:

1. settings
2. spectrum generation
3. GPU buffers
4. compute pipelines
5. render pipelines
6. displacement and normal texture outputs
7. reusable OceanRenderer component
   '@ | Set-Content -Path "src\ocean\webgpu\engine\README.md" -Encoding UTF8
