import * as THREE from "three";

let cachedTextures = null;

function waveHeight(u, v, layers) {
  let h = 0;

  for (let i = 0; i < layers.length; i += 1) {
    const layer = layers[i];

    const p = Math.PI * 2 * (layer.kx * u + layer.ky * v) + layer.phase;

    h += Math.sin(p) * layer.amp;
    h += Math.sin(p * 2.0 + layer.phase * 0.37) * layer.amp * 0.22;
  }

  return h;
}

function createNormalTexture(size, layers, strength = 3.0) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext("2d");
  const image = context.createImageData(size, size);
  const data = image.data;

  const eps = 1 / size;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const u = x / size;
      const v = y / size;

      const hL = waveHeight(u - eps, v, layers);
      const hR = waveHeight(u + eps, v, layers);
      const hD = waveHeight(u, v - eps, layers);
      const hU = waveHeight(u, v + eps, layers);

      const dx = (hR - hL) * strength;
      const dy = (hU - hD) * strength;

      const normal = new THREE.Vector3(-dx, -dy, 1).normalize();

      const index = (y * size + x) * 4;

      data[index + 0] = Math.floor((normal.x * 0.5 + 0.5) * 255);
      data[index + 1] = Math.floor((normal.y * 0.5 + 0.5) * 255);
      data[index + 2] = Math.floor((normal.z * 0.5 + 0.5) * 255);
      data[index + 3] = 255;
    }
  }

  context.putImageData(image, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;

  return texture;
}

function hash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

function smoothNoise(u, v, cells) {
  const x = u * cells;
  const y = v * cells;

  const ix = Math.floor(x);
  const iy = Math.floor(y);

  const fx = x - ix;
  const fy = y - iy;

  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);

  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);

  const ab = a * (1 - ux) + b * ux;
  const cd = c * (1 - ux) + d * ux;

  return ab * (1 - uy) + cd * uy;
}

function fbm(u, v) {
  let value = 0;
  let amp = 0.5;
  let cells = 4;

  for (let i = 0; i < 6; i += 1) {
    value += smoothNoise(u, v, cells) * amp;
    amp *= 0.5;
    cells *= 2;
  }

  return value;
}

function createFoamNoiseTexture(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext("2d");
  const image = context.createImageData(size, size);
  const data = image.data;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const u = x / size;
      const v = y / size;

      const n = fbm(u, v);
      const r = 1 - Math.abs(n * 2 - 1);
      const lace = Math.pow(Math.max(r, 0), 2.4);

      const streak = Math.sin((u * 18 + v * 5) * Math.PI * 2) * 0.5 + 0.5;

      const mask = Math.max(
        0,
        Math.min(1, lace * 0.75 + streak * 0.15 + n * 0.1),
      );

      const value = Math.floor(mask * 255);
      const index = (y * size + x) * 4;

      data[index + 0] = value;
      data[index + 1] = value;
      data[index + 2] = value;
      data[index + 3] = 255;
    }
  }

  context.putImageData(image, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;

  return texture;
}

export function createWaterTextures() {
  if (cachedTextures) return cachedTextures;

  const normalLayersA = [
    { kx: 1, ky: 0, amp: 1.0, phase: 0.1 },
    { kx: 2, ky: 1, amp: 0.48, phase: 1.4 },
    { kx: -3, ky: 2, amp: 0.25, phase: 2.7 },
    { kx: 5, ky: 3, amp: 0.12, phase: 4.2 },
    { kx: -8, ky: 5, amp: 0.055, phase: 0.9 },
  ];

  const normalLayersB = [
    { kx: 0, ky: 1, amp: 1.0, phase: 2.2 },
    { kx: -2, ky: 1, amp: 0.55, phase: 0.7 },
    { kx: 3, ky: 4, amp: 0.24, phase: 5.1 },
    { kx: -6, ky: 3, amp: 0.12, phase: 3.4 },
    { kx: 9, ky: -5, amp: 0.055, phase: 1.8 },
  ];

  cachedTextures = {
    normalA: createNormalTexture(512, normalLayersA, 3.2),
    normalB: createNormalTexture(512, normalLayersB, 2.7),
    foamNoise: createFoamNoiseTexture(512),
  };

  return cachedTextures;
}
