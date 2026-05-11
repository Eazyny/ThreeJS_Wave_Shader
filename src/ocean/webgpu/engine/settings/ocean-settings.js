export const FFT_SIZE = 128;
export const FFT_LOG_SIZE = 7;
export const OCEAN_SIZE = 360;

export const DEFAULT_OCEAN_SETTINGS = {
  windSpeed: 18,
  windDirection: 25,

  waveHeight: 1.28,
  choppiness: 1.04,
  spectrumStrength: 1.0,
  foamStrength: 0.24,

  bodyDetail: 0.88,

  sunDirection: {
    x: -0.48,
    y: 0.58,
    z: 0.66,
  },

  camera: {
    fov: 46,
    near: 0.1,
    far: 2600,
    target: {
      x: 0,
      y: 0.2,
      z: -44,
    },
    initialDistance: 88,
    initialYaw: 0.02,
    initialPitch: -0.12,
    heightOffset: 14,
  },

  spectrum: {
    seed: 1337,
    targetStd: 1.0,
    phillipsA: 1.0,
    windPower: 6,
    smallWaveDamp: 0.018,
    gravity: 9.81,
  },

  renderer: {
    maxDevicePixelRatio: 2,
    clearColor: {
      r: 0.53,
      g: 0.68,
      b: 0.78,
      a: 1,
    },
  },
};

export const ROUTE_LABELS = {
  webgpuProbe: "?webgpu=1",
  fftProbe: "?fft=1",
};
