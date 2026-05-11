import * as THREE from "three";
import {
  FFT_SIZE,
  OCEAN_SIZE,
  DEFAULT_OCEAN_SETTINGS,
} from "../settings/ocean-settings.js";

export function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianRandom(random) {
  let u = 0;
  let v = 0;

  while (u === 0) u = random();
  while (v === 0) v = random();

  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function createSpectrumModes({
  size = FFT_SIZE,
  oceanSize = OCEAN_SIZE,
  windSpeed = DEFAULT_OCEAN_SETTINGS.windSpeed,
  windDirection = DEFAULT_OCEAN_SETTINGS.windDirection,
  targetStd = DEFAULT_OCEAN_SETTINGS.spectrum.targetStd,
  seed = DEFAULT_OCEAN_SETTINGS.spectrum.seed,
  gravity = DEFAULT_OCEAN_SETTINGS.spectrum.gravity,
  phillipsA = DEFAULT_OCEAN_SETTINGS.spectrum.phillipsA,
  windPower = DEFAULT_OCEAN_SETTINGS.spectrum.windPower,
  smallWaveDamp = DEFAULT_OCEAN_SETTINGS.spectrum.smallWaveDamp,
} = {}) {
  const random = mulberry32(seed);

  const windAngle = THREE.MathUtils.degToRad(windDirection);
  const wind = new THREE.Vector2(Math.cos(windAngle), Math.sin(windAngle));

  const h0 = new Array(size * size);

  const largestWaveFromWind = Math.max(
    (windSpeed * windSpeed) / gravity,
    0.001,
  );

  let amplitudeEnergy = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const sx = x < size / 2 ? x : x - size;
      const sy = y < size / 2 ? y : y - size;

      const kx = (2 * Math.PI * sx) / oceanSize;
      const kz = (2 * Math.PI * sy) / oceanSize;

      const kLength = Math.sqrt(kx * kx + kz * kz);

      let real = 0;
      let imag = 0;
      let directionWeight = 0;
      let omega = 0;

      if (kLength > 0.00001) {
        const kDir = new THREE.Vector2(kx / kLength, kz / kLength);
        const alignment = kDir.dot(wind);

        const forward = Math.max(alignment, 0);
        const backward = Math.max(-alignment, 0);

        directionWeight =
          Math.pow(forward, windPower) + Math.pow(backward, 2) * 0.035;

        const phillips =
          phillipsA *
          (Math.exp(-1 / Math.pow(kLength * largestWaveFromWind, 2)) /
            Math.pow(kLength, 4)) *
          directionWeight *
          Math.exp(-kLength * kLength * smallWaveDamp * smallWaveDamp);

        const spectrumAmplitude = Math.sqrt(Math.max(phillips, 0) / 2);

        real = gaussianRandom(random) * spectrumAmplitude;
        imag = gaussianRandom(random) * spectrumAmplitude;

        omega = Math.sqrt(gravity * kLength);
      }

      const index = y * size + x;

      h0[index] = {
        kx,
        kz,
        real,
        imag,
        omega,
        directionWeight,
      };

      amplitudeEnergy += real * real + imag * imag;
    }
  }

  const currentStd = Math.sqrt(Math.max(amplitudeEnergy, 0.000001));
  const normalizeFactor = targetStd / currentStd;

  const modes = new Float32Array(size * size * 8);

  let pointer = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = y * size + x;

      const xNeg = (size - x) % size;
      const yNeg = (size - y) % size;
      const negIndex = yNeg * size + xNeg;

      const mode = h0[index];
      const negMode = h0[negIndex];

      const h0Real = mode.real * normalizeFactor;
      const h0Imag = mode.imag * normalizeFactor;

      const h0NegConjReal = negMode.real * normalizeFactor;
      const h0NegConjImag = -negMode.imag * normalizeFactor;

      modes[pointer + 0] = mode.kx;
      modes[pointer + 1] = mode.kz;
      modes[pointer + 2] = h0Real;
      modes[pointer + 3] = h0Imag;

      modes[pointer + 4] = h0NegConjReal;
      modes[pointer + 5] = h0NegConjImag;
      modes[pointer + 6] = mode.omega;
      modes[pointer + 7] = mode.directionWeight;

      pointer += 8;
    }
  }

  return modes;
}
