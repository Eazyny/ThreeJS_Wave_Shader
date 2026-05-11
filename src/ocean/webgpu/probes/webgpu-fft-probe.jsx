import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const FFT_SIZE = 128;
const FFT_LOG_SIZE = 7;
const OCEAN_SIZE = 360;

const computeSpectrumShader = /* wgsl */ `
struct SpectrumMode {
  data0: vec4f,
  data1: vec4f,
};

struct Params {
  time: f32,
  gridSize: f32,
  oceanSize: f32,
  windSpeed: f32,

  waveHeight: f32,
  choppiness: f32,
  spectrumStrength: f32,
  foamStrength: f32,

  bodyDetail: f32,
  windDirection: f32,
  padding0: f32,
  padding1: f32,

  sunDir: vec4f,
  cameraPos: vec4f,
  viewProjection: mat4x4f,
};

@group(0) @binding(0) var<storage, read_write> spectrumOut: array<vec4f>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> modes: array<SpectrumMode>;

const FFT_SIZE_U: u32 = ${FFT_SIZE}u;
const FFT_LOG_SIZE_U: u32 = ${FFT_LOG_SIZE}u;

fn complex_mul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

fn bit_reverse(v: u32) -> u32 {
  var x = v;
  var r = 0u;

  for (var i = 0u; i < FFT_LOG_SIZE_U; i = i + 1u) {
    r = (r << 1u) | (x & 1u);
    x = x >> 1u;
  }

  return r;
}

@compute @workgroup_size(256)
fn csMain(@builtin(global_invocation_id) globalId: vec3u) {
  let id = globalId.x;
  let total = FFT_SIZE_U * FFT_SIZE_U;

  if (id >= total) {
    return;
  }

  let x = id % FFT_SIZE_U;
  let y = id / FFT_SIZE_U;

  let mode = modes[id];

  let h0 = mode.data0.zw;
  let h0NegConj = mode.data1.xy;
  let omega = mode.data1.z;
  let directionWeight = mode.data1.w;

  let wt = omega * params.time;

  let expPos = vec2f(cos(wt), sin(wt));
  let expNeg = vec2f(cos(-wt), sin(-wt));

  var hkt = complex_mul(h0, expPos) + complex_mul(h0NegConj, expNeg);

  let directionalLift = mix(0.72, 1.18, clamp(directionWeight, 0.0, 1.0));
  hkt *= directionalLift;

  let rx = bit_reverse(x);
  let ry = bit_reverse(y);

  let outIndex = ry * FFT_SIZE_U + rx;

  spectrumOut[outIndex] = vec4f(hkt, 0.0, 0.0);
}
`;

const fftShader = /* wgsl */ `
struct FFTParams {
  stage: u32,
  direction: u32,
  size: u32,
  padding: u32,
};

@group(0) @binding(0) var<storage, read> inputBuffer: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<vec4f>;
@group(0) @binding(2) var<uniform> fftParams: FFTParams;

fn complex_mul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

@compute @workgroup_size(256)
fn csMain(@builtin(global_invocation_id) globalId: vec3u) {
  let pairId = globalId.x;
  let n = fftParams.size;
  let totalPairs = (n * n) / 2u;

  if (pairId >= totalPairs) {
    return;
  }

  let halfSize = 1u << fftParams.stage;
  let butterflySize = halfSize * 2u;

  var indexA = 0u;
  var indexB = 0u;

  let localPair = pairId % (n / 2u);
  let line = pairId / (n / 2u);

  let block = localPair / halfSize;
  let j = localPair % halfSize;

  if (fftParams.direction == 0u) {
    let y = line;
    let x0 = block * butterflySize + j;
    let x1 = x0 + halfSize;

    indexA = y * n + x0;
    indexB = y * n + x1;
  } else {
    let x = line;
    let y0 = block * butterflySize + j;
    let y1 = y0 + halfSize;

    indexA = y0 * n + x;
    indexB = y1 * n + x;
  }

  let a = inputBuffer[indexA].xy;
  let b = inputBuffer[indexB].xy;

  let angle = 6.283185307179586 * f32(j) / f32(butterflySize);
  let w = vec2f(cos(angle), sin(angle));

  let t = complex_mul(w, b);

  outputBuffer[indexA] = vec4f(a + t, 0.0, 0.0);
  outputBuffer[indexB] = vec4f(a - t, 0.0, 0.0);
}
`;

const sampleShader = /* wgsl */ `
struct Sample {
  position: vec4f,
  normal: vec4f,
};

struct Params {
  time: f32,
  gridSize: f32,
  oceanSize: f32,
  windSpeed: f32,

  waveHeight: f32,
  choppiness: f32,
  spectrumStrength: f32,
  foamStrength: f32,

  bodyDetail: f32,
  windDirection: f32,
  padding0: f32,
  padding1: f32,

  sunDir: vec4f,
  cameraPos: vec4f,
  viewProjection: mat4x4f,
};

@group(0) @binding(0) var<storage, read> fftValues: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(2) var<uniform> params: Params;

fn safe_index(x: i32, y: i32) -> u32 {
  let n = i32(params.gridSize);

  let xx = clamp(x, 0, n - 1);
  let yy = clamp(y, 0, n - 1);

  return u32(yy * n + xx);
}

fn height_at(x: i32, y: i32) -> f32 {
  let raw = fftValues[safe_index(x, y)].x;
  return raw * params.waveHeight * params.spectrumStrength;
}

@compute @workgroup_size(256)
fn csMain(@builtin(global_invocation_id) globalId: vec3u) {
  let id = globalId.x;
  let n = u32(params.gridSize);
  let total = n * n;

  if (id >= total) {
    return;
  }

  let x = id % n;
  let y = id / n;

  let xi = i32(x);
  let yi = i32(y);

  let fx = f32(x) / f32(n - 1u) - 0.5;
  let fy = f32(y) / f32(n - 1u) - 0.5;

  let spacing = params.oceanSize / f32(n - 1u);

  let h = height_at(xi, yi);

  let hL = height_at(xi - 1, yi);
  let hR = height_at(xi + 1, yi);
  let hD = height_at(xi, yi - 1);
  let hU = height_at(xi, yi + 1);

  let gradX = (hR - hL) / (2.0 * spacing);
  let gradZ = (hU - hD) / (2.0 * spacing);

  let normal = normalize(vec3f(
    -gradX,
    1.0,
    -gradZ
  ));

  let chopScale = params.choppiness * 5.5;

  let chop = vec2f(
    -gradX,
    -gradZ
  ) * chopScale;

  let baseX = fx * params.oceanSize;
  let baseZ = fy * params.oceanSize;

  samples[id].position = vec4f(
    baseX + chop.x,
    h,
    baseZ + chop.y,
    h
  );

  samples[id].normal = vec4f(
    normal,
    length(vec2f(gradX, gradZ))
  );
}
`;

const renderShader = /* wgsl */ `
struct Sample {
  position: vec4f,
  normal: vec4f,
};

struct Params {
  time: f32,
  gridSize: f32,
  oceanSize: f32,
  windSpeed: f32,

  waveHeight: f32,
  choppiness: f32,
  spectrumStrength: f32,
  foamStrength: f32,

  bodyDetail: f32,
  windDirection: f32,
  padding0: f32,
  padding1: f32,

  sunDir: vec4f,
  cameraPos: vec4f,
  viewProjection: mat4x4f,
};

struct VertexOut {
  @builtin(position) clipPosition: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) normal: vec3f,
  @location(2) height: f32,
  @location(3) slope: f32,
  @location(4) gradientStrength: f32,
};

@group(0) @binding(0) var<storage, read> samples: array<Sample>;
@group(0) @binding(1) var<uniform> params: Params;

fn hash(p: vec2f) -> f32 {
  let h = sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123;
  return fract(h);
}

fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);

  let a = hash(i);
  let b = hash(i + vec2f(1.0, 0.0));
  let c = hash(i + vec2f(0.0, 1.0));
  let d = hash(i + vec2f(1.0, 1.0));

  let u = f * f * (3.0 - 2.0 * f);

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(pIn: vec2f) -> f32 {
  var p = pIn;
  var value = 0.0;
  var amp = 0.5;

  for (var i = 0; i < 5; i = i + 1) {
    value += noise(p) * amp;
    p *= 2.0;
    amp *= 0.5;
  }

  return value;
}

fn schlick_fresnel(cosTheta: f32, ior: f32) -> f32 {
  let f0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

fn sample_sky(dirIn: vec3f) -> vec3f {
  let dir = normalize(dirIn);
  let sunDir = normalize(params.sunDir.xyz);

  let vertical = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

  let horizon = vec3f(0.82, 0.88, 0.90);
  let mid = vec3f(0.45, 0.68, 0.82);
  let top = vec3f(0.23, 0.42, 0.62);

  var sky = mix(horizon, mid, smoothstep(0.0, 0.55, vertical));
  sky = mix(sky, top, smoothstep(0.38, 1.0, vertical));

  let sunAmount = max(dot(dir, sunDir), 0.0);

  sky += vec3f(1.0, 0.88, 0.62) * pow(sunAmount, 4.0) * 0.14;
  sky += vec3f(1.0, 0.92, 0.72) * pow(sunAmount, 56.0) * 1.25;

  return sky;
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  let sample = samples[vertexIndex];

  var out: VertexOut;

  out.worldPosition = sample.position.xyz;
  out.normal = normalize(sample.normal.xyz);
  out.height = sample.position.w;
  out.slope = length(out.normal.xz);
  out.gradientStrength = sample.normal.w;
  out.clipPosition = params.viewProjection * vec4f(out.worldPosition, 1.0);

  return out;
}

@fragment
fn fsMain(input: VertexOut) -> @location(0) vec4f {
  let N = normalize(input.normal);
  let V = normalize(params.cameraPos.xyz - input.worldPosition);
  let L = normalize(params.sunDir.xyz);

  let ndv = clamp(dot(N, V), 0.0, 1.0);
  let ndlRaw = dot(N, L);
  let ndl = clamp(ndlRaw, 0.0, 1.0);
  let grazing = 1.0 - ndv;

  let height = input.height;
  let slope = input.slope;
  let gradientStrength = input.gradientStrength;

  let windAngle = radians(params.windDirection);
  let wind = normalize(vec2f(cos(windAngle), sin(windAngle)));
  let windSide = vec2f(-wind.y, wind.x);

  let alongWind = dot(input.worldPosition.xz, wind);
  let acrossWind = dot(input.worldPosition.xz, windSide);

  let longVariation = fbm(input.worldPosition.xz * 0.010 + wind * params.time * 0.006);
  let mediumVariation = fbm(input.worldPosition.xz * 0.038 - wind * params.time * 0.018);
  let windStreaks = fbm(vec2f(acrossWind * 0.045, alongWind * 0.010 + params.time * 0.018));
  let detailVariation = fbm(input.worldPosition.xz * 0.12 + vec2f(params.time * 0.035, -params.time * 0.018));

  let variation =
    longVariation * 0.42 +
    mediumVariation * 0.28 +
    windStreaks * 0.22 +
    detailVariation * 0.08;

  let deepColor = vec3f(0.006, 0.105, 0.155);
  let midColor = vec3f(0.014, 0.335, 0.435);
  let shallowColor = vec3f(0.065, 0.68, 0.78);

  var body = mix(deepColor, midColor, smoothstep(-1.45, 0.35, height));
  body = mix(body, shallowColor, smoothstep(0.12, 1.35, height) * 0.17);

  let sunSide = smoothstep(-0.28, 0.82, ndlRaw);
  let shadowSide = 1.0 - sunSide;

  let trough = smoothstep(0.12, 1.20, -height);
  let crest = smoothstep(0.15, 1.22, height);
  let steepCrest = smoothstep(0.16, 0.50, slope) * crest;
  let waveEnergy = smoothstep(0.10, 1.2, gradientStrength);

  let bodyDetail = params.bodyDetail;

  body *= mix(1.0, mix(0.66, 1.24, variation), bodyDetail);
  body *= 1.0 - shadowSide * 0.27;
  body *= 1.0 - trough * 0.24;
  body += shallowColor * crest * 0.07;
  body += shallowColor * sunSide * 0.044;

  let absorption = vec3f(2.18, 1.02, 0.28);

  var opticalDepth = mix(1.08, 8.4, grazing);
  opticalDepth += trough * 1.24;
  opticalDepth += shadowSide * 0.82;
  opticalDepth += (1.0 - variation) * 0.78 * bodyDetail;

  let transmittance = exp(-absorption * opticalDepth);
  let scatter = mix(midColor, shallowColor, 0.33);
  let transmitted = body * transmittance + scatter * (1.0 - transmittance) * 0.52;

  let R = reflect(-V, N);
  let reflectedSky = sample_sky(R);

  let fresnel = clamp(schlick_fresnel(ndv, 1.333) * 1.28, 0.0, 1.0);
  let reflectionAmount = clamp(fresnel * 0.90, 0.02, 0.94);

  let H = normalize(L + V);
  let nh = max(dot(N, H), 0.0);

  let glitterNoise = fbm(input.worldPosition.xz * 0.22 + wind * params.time * 0.052);
  let glitterMask = smoothstep(0.55, 0.94, glitterNoise + slope * 0.22 + waveEnergy * 0.05);

  let spec =
    pow(nh, 30.0) * 0.05 +
    pow(nh, 160.0) * 0.50 +
    pow(nh, 620.0) * 1.95;

  let sunSpec = spec * glitterMask * ndl * 2.85;

  var color = mix(transmitted, reflectedSky, reflectionAmount);
  color += vec3f(1.0, 0.90, 0.70) * sunSpec;

  let foamNoiseA = fbm(vec2f(acrossWind * 0.085, alongWind * 0.018 + params.time * 0.035));
  let foamNoiseB = fbm(input.worldPosition.xz * 0.105 + wind * params.time * 0.03);

  let foamSource = smoothstep(
    0.34,
    1.12,
    steepCrest + trough * 0.18 + grazing * 0.12 + waveEnergy * 0.20
  );

  let foam =
    smoothstep(0.86, 0.98, foamNoiseA * 0.65 + foamNoiseB * 0.35) *
    foamSource *
    params.foamStrength;

  color = mix(color, vec3f(0.92, 0.98, 1.0), clamp(foam, 0.0, 1.0));

  let distanceToCamera = length(params.cameraPos.xyz - input.worldPosition);
  let fogAmount = smoothstep(340.0, 1180.0, distanceToCamera);
  color = mix(color, vec3f(0.70, 0.82, 0.88), fogAmount * 0.15);

  color = color / (color + vec3f(1.0));
  color = pow(color, vec3f(0.88));

  return vec4f(color, 1.0);
}
`;

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(random) {
  let u = 0;
  let v = 0;

  while (u === 0) u = random();
  while (v === 0) v = random();

  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function createSpectrumModes({
  size,
  oceanSize,
  windSpeed,
  windDirection,
  targetStd = 1.0,
}) {
  const random = mulberry32(1337);

  const g = 9.81;
  const windAngle = THREE.MathUtils.degToRad(windDirection);
  const wind = new THREE.Vector2(Math.cos(windAngle), Math.sin(windAngle));

  const h0 = new Array(size * size);

  const L = Math.max((windSpeed * windSpeed) / g, 0.001);
  const smallWaveDamp = 0.018;
  const phillipsA = 1.0;
  const windPower = 6;

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
          Math.pow(forward, windPower) +
          Math.pow(backward, 2) * 0.035;

        const phillips =
          phillipsA *
          (Math.exp(-1 / Math.pow(kLength * L, 2)) /
            Math.pow(kLength, 4)) *
          directionWeight *
          Math.exp(-kLength * kLength * smallWaveDamp * smallWaveDamp);

        const spectrumAmplitude = Math.sqrt(Math.max(phillips, 0) / 2);

        real = gaussianRandom(random) * spectrumAmplitude;
        imag = gaussianRandom(random) * spectrumAmplitude;

        omega = Math.sqrt(g * kLength);
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

function createGridIndices(size) {
  const cells = size - 1;
  const indices = new Uint32Array(cells * cells * 6);

  let pointer = 0;

  for (let z = 0; z < cells; z += 1) {
    for (let x = 0; x < cells; x += 1) {
      const a = z * size + x;
      const b = a + 1;
      const c = a + size;
      const d = c + 1;

      indices[pointer++] = a;
      indices[pointer++] = c;
      indices[pointer++] = b;

      indices[pointer++] = b;
      indices[pointer++] = c;
      indices[pointer++] = d;
    }
  }

  return indices;
}

function createFFTParamBuffer(device, stage, direction, size) {
  const values = new Uint32Array([stage, direction, size, 0]);

  const buffer = device.createBuffer({
    size: values.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(buffer, 0, values);

  return buffer;
}

export default function WebGPUFFTProbe() {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const frameRef = useRef(null);

  const controlRef = useRef({
    yaw: 0.02,
    pitch: -0.12,
    distance: 88,
    dragging: false,
    lastX: 0,
    lastY: 0,
  });

  const [status, setStatus] = useState(
    'Initializing GPU IFFT / Stockham foundation...'
  );

  const [details, setDetails] = useState(
    'Preparing h(k,t), horizontal FFT, vertical FFT, and sample buffers.'
  );

  useEffect(() => {
    let device = null;
    let context = null;
    let depthTexture = null;
    let resizeObserver = null;
    let destroyed = false;

    async function init() {
      if (!canvasRef.current || !wrapRef.current) return;

      if (!navigator.gpu) {
        setStatus('Native WebGPU was not detected.');
        setDetails('This probe needs native navigator.gpu, not WebGL fallback.');
        return;
      }

      const adapter = await navigator.gpu.requestAdapter();

      if (!adapter) {
        setStatus('WebGPU adapter request failed.');
        setDetails('Browser found navigator.gpu, but no GPU adapter was returned.');
        return;
      }

      device = await adapter.requestDevice();

      if (destroyed) return;

      const canvas = canvasRef.current;
      context = canvas.getContext('webgpu');

      const format = navigator.gpu.getPreferredCanvasFormat();

      const vertexCount = FFT_SIZE * FFT_SIZE;
      const complexBufferSize = vertexCount * 4 * 4;
      const sampleBufferSize = vertexCount * 8 * 4;

      const fftBufferA = device.createBuffer({
        size: complexBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const fftBufferB = device.createBuffer({
        size: complexBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const sampleBuffer = device.createBuffer({
        size: sampleBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const uniformBuffer = device.createBuffer({
        size: 36 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const spectrumModes = createSpectrumModes({
        size: FFT_SIZE,
        oceanSize: OCEAN_SIZE,
        windSpeed: 18,
        windDirection: 25,
        targetStd: 1.0,
      });

      const spectrumBuffer = device.createBuffer({
        size: spectrumModes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      device.queue.writeBuffer(spectrumBuffer, 0, spectrumModes);

      const indices = createGridIndices(FFT_SIZE);

      const indexBuffer = device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });

      device.queue.writeBuffer(indexBuffer, 0, indices);

      const spectrumModule = device.createShaderModule({
        label: 'GPU IFFT spectrum shader',
        code: computeSpectrumShader,
      });

      const fftModule = device.createShaderModule({
        label: 'GPU IFFT Stockham shader',
        code: fftShader,
      });

      const sampleModule = device.createShaderModule({
        label: 'GPU IFFT sample shader',
        code: sampleShader,
      });

      const renderModule = device.createShaderModule({
        label: 'GPU IFFT render shader',
        code: renderShader,
      });

      const spectrumPipeline = device.createComputePipeline({
        label: 'GPU IFFT spectrum pipeline',
        layout: 'auto',
        compute: {
          module: spectrumModule,
          entryPoint: 'csMain',
        },
      });

      const fftPipeline = device.createComputePipeline({
        label: 'GPU IFFT Stockham pipeline',
        layout: 'auto',
        compute: {
          module: fftModule,
          entryPoint: 'csMain',
        },
      });

      const samplePipeline = device.createComputePipeline({
        label: 'GPU IFFT sample pipeline',
        layout: 'auto',
        compute: {
          module: sampleModule,
          entryPoint: 'csMain',
        },
      });

      const renderPipeline = device.createRenderPipeline({
        label: 'GPU IFFT render pipeline',
        layout: 'auto',
        vertex: {
          module: renderModule,
          entryPoint: 'vsMain',
        },
        fragment: {
          module: renderModule,
          entryPoint: 'fsMain',
          targets: [{ format }],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',
        },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less',
        },
      });

      const spectrumBindGroup = device.createBindGroup({
        label: 'GPU IFFT spectrum bind group',
        layout: spectrumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fftBufferA } },
          { binding: 1, resource: { buffer: uniformBuffer } },
          { binding: 2, resource: { buffer: spectrumBuffer } },
        ],
      });

      const fftLayout = fftPipeline.getBindGroupLayout(0);

      const rowBindGroups = [];
      const columnBindGroups = [];

      for (let stage = 0; stage < FFT_LOG_SIZE; stage += 1) {
        const rowParamBuffer = createFFTParamBuffer(device, stage, 0, FFT_SIZE);
        const columnParamBuffer = createFFTParamBuffer(device, stage, 1, FFT_SIZE);

        const rowInput = stage % 2 === 0 ? fftBufferA : fftBufferB;
        const rowOutput = stage % 2 === 0 ? fftBufferB : fftBufferA;

        rowBindGroups.push(
          device.createBindGroup({
            label: `GPU IFFT row stage ${stage}`,
            layout: fftLayout,
            entries: [
              { binding: 0, resource: { buffer: rowInput } },
              { binding: 1, resource: { buffer: rowOutput } },
              { binding: 2, resource: { buffer: rowParamBuffer } },
            ],
          })
        );

        const columnInput = stage % 2 === 0 ? fftBufferB : fftBufferA;
        const columnOutput = stage % 2 === 0 ? fftBufferA : fftBufferB;

        columnBindGroups.push(
          device.createBindGroup({
            label: `GPU IFFT column stage ${stage}`,
            layout: fftLayout,
            entries: [
              { binding: 0, resource: { buffer: columnInput } },
              { binding: 1, resource: { buffer: columnOutput } },
              { binding: 2, resource: { buffer: columnParamBuffer } },
            ],
          })
        );
      }

      const sampleBindGroup = device.createBindGroup({
        label: 'GPU IFFT sample bind group',
        layout: samplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fftBufferA } },
          { binding: 1, resource: { buffer: sampleBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const renderBindGroup = device.createBindGroup({
        label: 'GPU IFFT render bind group',
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sampleBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
      });

      const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 2600);
      const target = new THREE.Vector3(0, 0.2, -44);
      const viewProjection = new THREE.Matrix4();
      const uniformArray = new Float32Array(36);

      function resize() {
        if (!wrapRef.current || !canvasRef.current || !context || !device) {
          return;
        }

        const width = Math.max(1, wrapRef.current.clientWidth);
        const height = Math.max(1, wrapRef.current.clientHeight);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);

        context.configure({
          device,
          format,
          alphaMode: 'opaque',
        });

        if (depthTexture) {
          depthTexture.destroy();
        }

        depthTexture = device.createTexture({
          size: [canvas.width, canvas.height],
          format: 'depth24plus',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        camera.aspect = canvas.width / Math.max(canvas.height, 1);
        camera.updateProjectionMatrix();
      }

      resizeObserver = new ResizeObserver(resize);
      resizeObserver.observe(wrapRef.current);
      resize();

      const start = performance.now();

      function updateCamera() {
        const controls = controlRef.current;
        const horizontal = Math.cos(controls.pitch) * controls.distance;

        camera.position.set(
          Math.sin(controls.yaw) * horizontal,
          Math.sin(controls.pitch) * controls.distance + 14,
          Math.cos(controls.yaw) * horizontal
        );

        camera.lookAt(target);
        camera.updateMatrixWorld(true);

        viewProjection.multiplyMatrices(
          camera.projectionMatrix,
          camera.matrixWorldInverse
        );
      }

      function writeUniforms(timeSeconds) {
        updateCamera();

        uniformArray[0] = timeSeconds + 60.0;
        uniformArray[1] = FFT_SIZE;
        uniformArray[2] = OCEAN_SIZE;
        uniformArray[3] = 18.0;

        uniformArray[4] = 1.28;
        uniformArray[5] = 1.04;
        uniformArray[6] = 1.0;
        uniformArray[7] = 0.24;

        uniformArray[8] = 0.88;
        uniformArray[9] = 25.0;
        uniformArray[10] = 0.0;
        uniformArray[11] = 0.0;

        uniformArray[12] = -0.48;
        uniformArray[13] = 0.58;
        uniformArray[14] = 0.66;
        uniformArray[15] = 0.0;

        uniformArray[16] = camera.position.x;
        uniformArray[17] = camera.position.y;
        uniformArray[18] = camera.position.z;
        uniformArray[19] = 0.0;

        uniformArray.set(viewProjection.elements, 20);

        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
      }

      function frame() {
        if (destroyed || !device || !context || !depthTexture) return;

        const timeSeconds = (performance.now() - start) * 0.001;

        writeUniforms(timeSeconds);

        const encoder = device.createCommandEncoder();

        const spectrumPass = encoder.beginComputePass({
          label: 'GPU IFFT h(k,t) spectrum pass',
        });

        spectrumPass.setPipeline(spectrumPipeline);
        spectrumPass.setBindGroup(0, spectrumBindGroup);
        spectrumPass.dispatchWorkgroups(Math.ceil(vertexCount / 256));
        spectrumPass.end();

        for (let stage = 0; stage < FFT_LOG_SIZE; stage += 1) {
          const rowPass = encoder.beginComputePass({
            label: `GPU IFFT horizontal stage ${stage}`,
          });

          rowPass.setPipeline(fftPipeline);
          rowPass.setBindGroup(0, rowBindGroups[stage]);
          rowPass.dispatchWorkgroups(Math.ceil(vertexCount / 2 / 256));
          rowPass.end();
        }

        for (let stage = 0; stage < FFT_LOG_SIZE; stage += 1) {
          const columnPass = encoder.beginComputePass({
            label: `GPU IFFT vertical stage ${stage}`,
          });

          columnPass.setPipeline(fftPipeline);
          columnPass.setBindGroup(0, columnBindGroups[stage]);
          columnPass.dispatchWorkgroups(Math.ceil(vertexCount / 2 / 256));
          columnPass.end();
        }

        const samplePass = encoder.beginComputePass({
          label: 'GPU IFFT displacement sample pass',
        });

        samplePass.setPipeline(samplePipeline);
        samplePass.setBindGroup(0, sampleBindGroup);
        samplePass.dispatchWorkgroups(Math.ceil(vertexCount / 256));
        samplePass.end();

        const colorView = context.getCurrentTexture().createView();
        const depthView = depthTexture.createView();

        const renderPass = encoder.beginRenderPass({
          label: 'GPU IFFT ocean render pass',
          colorAttachments: [
            {
              view: colorView,
              clearValue: { r: 0.53, g: 0.68, b: 0.78, a: 1 },
              loadOp: 'clear',
              storeOp: 'store',
            },
          ],
          depthStencilAttachment: {
            view: depthView,
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
          },
        });

        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.setIndexBuffer(indexBuffer, 'uint32');
        renderPass.drawIndexed(indices.length);
        renderPass.end();

        device.queue.submit([encoder.finish()]);

        frameRef.current = requestAnimationFrame(frame);
      }

      frame();

      setStatus('v0.18-D GPU IFFT / Stockham Foundation is running.');
      setDetails(
        `${FFT_SIZE}x${FFT_SIZE} h(k,t) spectrum → horizontal FFT → vertical FFT → displacement/normal sample buffer. Next: displacement texture + normal texture output.`
      );
    }

    init().catch((error) => {
      console.error(error);
      setStatus('WebGPU GPU IFFT foundation failed.');
      setDetails(error?.message || 'Unknown WebGPU error.');
    });

    return () => {
      destroyed = true;

      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }

      if (resizeObserver) {
        resizeObserver.disconnect();
      }

      if (depthTexture) {
        depthTexture.destroy();
      }

      if (device) {
        device.destroy?.();
      }
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function onPointerDown(event) {
      controlRef.current.dragging = true;
      controlRef.current.lastX = event.clientX;
      controlRef.current.lastY = event.clientY;
      canvas.setPointerCapture?.(event.pointerId);
    }

    function onPointerMove(event) {
      const controls = controlRef.current;
      if (!controls.dragging) return;

      const dx = event.clientX - controls.lastX;
      const dy = event.clientY - controls.lastY;

      controls.lastX = event.clientX;
      controls.lastY = event.clientY;

      controls.yaw -= dx * 0.006;
      controls.pitch += dy * 0.004;
      controls.pitch = Math.max(-0.5, Math.min(0.35, controls.pitch));
    }

    function onPointerUp(event) {
      controlRef.current.dragging = false;
      canvas.releasePointerCapture?.(event.pointerId);
    }

    function onWheel(event) {
      event.preventDefault();

      const controls = controlRef.current;
      controls.distance += event.deltaY * 0.04;
      controls.distance = Math.max(24, Math.min(240, controls.distance));
    }

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointerleave', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
    };
  }, []);

  return (
    <section
      ref={wrapRef}
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'hidden',
        background: '#06101a',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          cursor: 'grab',
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 28,
          bottom: 28,
          width: 'min(700px, calc(100vw - 56px))',
          padding: '18px 20px',
          borderRadius: 18,
          border: '1px solid rgba(160, 230, 255, 0.25)',
          background:
            'linear-gradient(135deg, rgba(2, 14, 24, 0.72), rgba(5, 35, 50, 0.42))',
          boxShadow: '0 18px 70px rgba(0,0,0,0.3)',
          backdropFilter: 'blur(14px)',
          color: 'white',
          fontFamily:
            'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
          pointerEvents: 'none',
        }}
      >
        <p
          style={{
            margin: '0 0 8px',
            fontSize: 12,
            letterSpacing: '0.22em',
            textTransform: 'uppercase',
            color: 'rgba(205, 244, 255, 0.9)',
          }}
        >
          OceanShader Pro / v0.18-D
        </p>

        <h1
          style={{
            margin: 0,
            fontSize: 'clamp(28px, 4vw, 48px)',
            lineHeight: 0.96,
            letterSpacing: '-0.055em',
          }}
        >
          GPU IFFT / Stockham Foundation
        </h1>

        <p
          style={{
            margin: '12px 0 0',
            fontSize: 14,
            lineHeight: 1.65,
            color: 'rgba(235, 250, 255, 0.86)',
          }}
        >
          {status}
        </p>

        <p
          style={{
            margin: '8px 0 0',
            fontSize: 13,
            lineHeight: 1.55,
            color: 'rgba(190, 245, 255, 0.86)',
          }}
        >
          {details}
        </p>
      </div>
    </section>
  );
}