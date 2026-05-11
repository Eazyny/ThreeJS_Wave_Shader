import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  FFT_SIZE,
  FFT_LOG_SIZE,
  OCEAN_SIZE,
  DEFAULT_OCEAN_SETTINGS,
} from '../engine/settings/ocean-settings.js';
import { createSpectrumModes } from '../engine/utils/spectrum.js';
import { createGridIndices } from '../engine/utils/grid.js';
import {
  createAndUploadStorageBuffer,
  createFFTParamBuffer,
  createIndexBuffer,
  createStorageBuffer,
  createUniformBuffer,
} from '../engine/buffers/gpu-buffers.js';

const SETTINGS = DEFAULT_OCEAN_SETTINGS;
const FIELD_NAMES = ['height', 'gradientX', 'gradientZ', 'displacementX', 'displacementZ'];

const spectrumFieldsShader = /* wgsl */ `
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

@group(0) @binding(0) var<storage, read_write> heightSpectrumOut: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> gradientXSpectrumOut: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> gradientZSpectrumOut: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> displacementXSpectrumOut: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> displacementZSpectrumOut: array<vec4f>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> modes: array<SpectrumMode>;

const FFT_SIZE_U: u32 = ${FFT_SIZE}u;
const FFT_LOG_SIZE_U: u32 = ${FFT_LOG_SIZE}u;

fn complex_mul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

fn complex_mul_i_scale(value: vec2f, scale: f32) -> vec2f {
  return vec2f(-value.y * scale, value.x * scale);
}

fn complex_mul_negative_i_scale(value: vec2f, scale: f32) -> vec2f {
  return vec2f(value.y * scale, -value.x * scale);
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

  let k = mode.data0.xy;
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

  let kLength = max(length(k), 0.0001);
  let kDir = k / kLength;

  /*
    Strict Tessendorf fields:
    height: h(k,t)
    gradient X: i * kx * h(k,t)
    gradient Z: i * kz * h(k,t)
    displacement X: -i * kx/|k| * h(k,t)
    displacement Z: -i * kz/|k| * h(k,t)
  */
  let heightSpectrum = hkt;
  let gradientXSpectrum = complex_mul_i_scale(hkt, k.x);
  let gradientZSpectrum = complex_mul_i_scale(hkt, k.y);
  let displacementXSpectrum = complex_mul_negative_i_scale(hkt, kDir.x);
  let displacementZSpectrum = complex_mul_negative_i_scale(hkt, kDir.y);

  let rx = bit_reverse(x);
  let ry = bit_reverse(y);

  let outIndex = ry * FFT_SIZE_U + rx;

  heightSpectrumOut[outIndex] = vec4f(heightSpectrum, 0.0, 0.0);
  gradientXSpectrumOut[outIndex] = vec4f(gradientXSpectrum, 0.0, 0.0);
  gradientZSpectrumOut[outIndex] = vec4f(gradientZSpectrum, 0.0, 0.0);
  displacementXSpectrumOut[outIndex] = vec4f(displacementXSpectrum, 0.0, 0.0);
  displacementZSpectrumOut[outIndex] = vec4f(displacementZSpectrum, 0.0, 0.0);
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

@group(0) @binding(0) var<storage, read> heightValues: array<vec4f>;
@group(0) @binding(1) var<storage, read> gradientXValues: array<vec4f>;
@group(0) @binding(2) var<storage, read> gradientZValues: array<vec4f>;
@group(0) @binding(3) var<storage, read> displacementXValues: array<vec4f>;
@group(0) @binding(4) var<storage, read> displacementZValues: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(6) var<uniform> params: Params;

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

  let fx = f32(x) / f32(n - 1u) - 0.5;
  let fy = f32(y) / f32(n - 1u) - 0.5;

  let checker = select(-1.0, 1.0, ((x + y) & 1u) == 0u);

  let heightScale = params.waveHeight * params.spectrumStrength;
  let chopScale = params.choppiness * params.waveHeight * params.spectrumStrength;

  let height = heightValues[id].x * checker * heightScale;

  let gradientX = gradientXValues[id].x * checker * heightScale;
  let gradientZ = gradientZValues[id].x * checker * heightScale;

  let displacementX = displacementXValues[id].x * checker * chopScale;
  let displacementZ = displacementZValues[id].x * checker * chopScale;

  let baseX = fx * params.oceanSize;
  let baseZ = fy * params.oceanSize;

  let normal = normalize(vec3f(
    -gradientX,
    1.0,
    -gradientZ
  ));

  samples[id].position = vec4f(
    baseX + displacementX,
    height,
    baseZ + displacementZ,
    height
  );

  samples[id].normal = vec4f(
    normal,
    length(vec2f(gradientX, gradientZ))
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

  let horizon = vec3f(0.78, 0.86, 0.90);
  let mid = vec3f(0.38, 0.60, 0.75);
  let top = vec3f(0.18, 0.34, 0.52);

  var sky = mix(horizon, mid, smoothstep(0.0, 0.55, vertical));
  sky = mix(sky, top, smoothstep(0.38, 1.0, vertical));

  let sunAmount = max(dot(dir, sunDir), 0.0);

  sky += vec3f(1.0, 0.86, 0.58) * pow(sunAmount, 5.0) * 0.06;
  sky += vec3f(1.0, 0.94, 0.76) * pow(sunAmount, 96.0) * 1.65;

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

  let height = input.height;
  let slope = input.slope;
  let gradientStrength = input.gradientStrength;

  let ndv = clamp(dot(N, V), 0.0, 1.0);
  let ndlRaw = dot(N, L);
  let ndl = clamp(ndlRaw, 0.0, 1.0);
  let grazing = 1.0 - ndv;

  let windAngle = radians(params.windDirection);
  let wind = normalize(vec2f(cos(windAngle), sin(windAngle)));
  let windSide = vec2f(-wind.y, wind.x);

  let alongWind = dot(input.worldPosition.xz, wind);
  let acrossWind = dot(input.worldPosition.xz, windSide);

  let longVariation = fbm(input.worldPosition.xz * 0.010 + wind * params.time * 0.006);
  let mediumVariation = fbm(input.worldPosition.xz * 0.035 - wind * params.time * 0.016);
  let windStreaks = fbm(vec2f(acrossWind * 0.045, alongWind * 0.010 + params.time * 0.018));

  let variation =
    longVariation * 0.42 +
    mediumVariation * 0.30 +
    windStreaks * 0.28;

  let waveEnergy = smoothstep(0.08, 1.65, gradientStrength);

  let deepColor = vec3f(0.0025, 0.060, 0.092);
  let midColor = vec3f(0.008, 0.230, 0.320);
  let shallowColor = vec3f(0.042, 0.48, 0.58);

  var body = mix(deepColor, midColor, smoothstep(-1.50, 0.30, height));
  body = mix(body, shallowColor, smoothstep(0.20, 1.40, height) * 0.10);

  let sunSide = smoothstep(-0.35, 0.82, ndlRaw);
  let shadowSide = 1.0 - sunSide;

  let trough = smoothstep(0.12, 1.20, -height);
  let crest = smoothstep(0.15, 1.22, height);
  let steepCrest = smoothstep(0.20, 0.58, slope) * crest;

  body *= mix(1.0, mix(0.70, 1.08, variation), params.bodyDetail);
  body *= 1.0 - shadowSide * 0.30;
  body *= 1.0 - trough * 0.26;
  body += shallowColor * crest * 0.035;
  body += shallowColor * sunSide * 0.018;

  let absorption = vec3f(2.40, 1.16, 0.32);

  var opticalDepth = mix(1.20, 9.8, grazing);
  opticalDepth += trough * 1.35;
  opticalDepth += shadowSide * 0.95;
  opticalDepth += (1.0 - variation) * 0.82 * params.bodyDetail;

  let transmittance = exp(-absorption * opticalDepth);
  let scatter = mix(midColor, shallowColor, 0.20);
  let transmitted = body * transmittance + scatter * (1.0 - transmittance) * 0.38;

  let R = reflect(-V, N);
  let reflectedSky = sample_sky(R);

  let fresnel = clamp(schlick_fresnel(ndv, 1.333) * 1.50, 0.0, 1.0);
  let reflectionAmount = clamp(fresnel * 1.04 + waveEnergy * 0.025, 0.03, 0.94);

  let sunReflect = max(dot(reflect(-L, N), V), 0.0);

  let glintNoiseFine = fbm(input.worldPosition.xz * 1.65 + wind * params.time * 0.20);
  let glintNoiseNeedle = fbm(vec2f(acrossWind * 0.30, alongWind * 0.055 + params.time * 0.12));
  let glintNoiseSpark = noise(input.worldPosition.xz * 2.75 + windSide * params.time * 0.28);

  let tinyGlintMask = smoothstep(
    0.82,
    0.995,
    glintNoiseFine * 0.42 +
      glintNoiseNeedle * 0.42 +
      glintNoiseSpark * 0.16 +
      slope * 0.08 +
      waveEnergy * 0.05
  );

  let directionalStretch = smoothstep(
    0.20,
    1.0,
    dot(normalize(vec2f(N.x, N.z) + vec2f(0.0001, 0.0001)), wind) * 0.5 + 0.5
  );

  let wideSpec = pow(sunReflect, 64.0) * 0.08;
  let tightSpec = pow(sunReflect, 260.0) * 0.72;
  let razorSpec = pow(sunReflect, 980.0) * 3.15;

  let sunGlint =
    (wideSpec + tightSpec + razorSpec) *
    tinyGlintMask *
    mix(0.42, 1.0, directionalStretch) *
    mix(0.50, 1.20, waveEnergy) *
    max(ndl, 0.0);

  var color = mix(transmitted * 0.84, reflectedSky * 0.58, reflectionAmount);

  color *= mix(0.76, 1.0, grazing);
  color += vec3f(1.0, 0.96, 0.84) * sunGlint;

  let foamLineNoise = fbm(vec2f(acrossWind * 0.18, alongWind * 0.035 + params.time * 0.050));
  let foamFineNoise = noise(input.worldPosition.xz * 1.15 + wind * params.time * 0.08);

  let crestOnlyFoamSource = smoothstep(
    0.42,
    1.14,
    steepCrest + waveEnergy * 0.26 + crest * 0.20
  );

  let foam =
    smoothstep(0.88, 0.99, foamLineNoise * 0.62 + foamFineNoise * 0.38) *
    crestOnlyFoamSource *
    params.foamStrength *
    0.42;

  color = mix(color, vec3f(0.88, 0.96, 1.0), clamp(foam, 0.0, 1.0));

  let distanceToCamera = length(params.cameraPos.xyz - input.worldPosition);
  let fogAmount = smoothstep(460.0, 1380.0, distanceToCamera);
  color = mix(color, vec3f(0.64, 0.76, 0.82), fogAmount * 0.08);

  color = color / (color + vec3f(1.0));
  color = pow(color, vec3f(0.84));

  return vec4f(color, 1.0);
}
`;

function createFFTFieldBuffers(device, complexBufferSize) {
  const fields = {};

  for (const fieldName of FIELD_NAMES) {
    fields[fieldName] = {
      a: createStorageBuffer(device, complexBufferSize, `${fieldName} FFT Buffer A`),
      b: createStorageBuffer(device, complexBufferSize, `${fieldName} FFT Buffer B`),
    };
  }

  return fields;
}

function createFieldFFTBindGroups(device, fftLayout, fields) {
  const rowBindGroups = {};
  const columnBindGroups = {};

  for (const fieldName of FIELD_NAMES) {
    rowBindGroups[fieldName] = [];
    columnBindGroups[fieldName] = [];

    const field = fields[fieldName];

    for (let stage = 0; stage < FFT_LOG_SIZE; stage += 1) {
      const rowParamBuffer = createFFTParamBuffer(device, stage, 0, FFT_SIZE);
      const columnParamBuffer = createFFTParamBuffer(device, stage, 1, FFT_SIZE);

      const rowInput = stage % 2 === 0 ? field.a : field.b;
      const rowOutput = stage % 2 === 0 ? field.b : field.a;

      rowBindGroups[fieldName].push(
        device.createBindGroup({
          label: `${fieldName} horizontal FFT stage ${stage}`,
          layout: fftLayout,
          entries: [
            { binding: 0, resource: { buffer: rowInput } },
            { binding: 1, resource: { buffer: rowOutput } },
            { binding: 2, resource: { buffer: rowParamBuffer } },
          ],
        })
      );

      const columnInput = stage % 2 === 0 ? field.b : field.a;
      const columnOutput = stage % 2 === 0 ? field.a : field.b;

      columnBindGroups[fieldName].push(
        device.createBindGroup({
          label: `${fieldName} vertical FFT stage ${stage}`,
          layout: fftLayout,
          entries: [
            { binding: 0, resource: { buffer: columnInput } },
            { binding: 1, resource: { buffer: columnOutput } },
            { binding: 2, resource: { buffer: columnParamBuffer } },
          ],
        })
      );
    }
  }

  return { rowBindGroups, columnBindGroups };
}

export default function WebGPUFFTProbe() {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const frameRef = useRef(null);

  const controlRef = useRef({
    yaw: SETTINGS.camera.initialYaw,
    pitch: SETTINGS.camera.initialPitch,
    distance: SETTINGS.camera.initialDistance,
    dragging: false,
    lastX: 0,
    lastY: 0,
  });

  const [status, setStatus] = useState(
    'Initializing strict Tessendorf IFFT pipeline...'
  );

  const [details, setDetails] = useState(
    'Computing height, gradients, and horizontal displacements as separate IFFT fields.'
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
        setDetails('This route needs native navigator.gpu, not WebGL fallback.');
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

      const fftFields = createFFTFieldBuffers(device, complexBufferSize);

      const sampleBuffer = createStorageBuffer(
        device,
        sampleBufferSize,
        'Strict Tessendorf Ocean Sample Buffer'
      );

      const uniformBuffer = createUniformBuffer(
        device,
        36 * 4,
        'Strict Tessendorf Ocean Uniform Buffer'
      );

      const spectrumModes = createSpectrumModes({
        size: FFT_SIZE,
        oceanSize: OCEAN_SIZE,
        windSpeed: SETTINGS.windSpeed,
        windDirection: SETTINGS.windDirection,
        targetStd: SETTINGS.spectrum.targetStd,
        seed: SETTINGS.spectrum.seed,
        gravity: SETTINGS.spectrum.gravity,
        phillipsA: SETTINGS.spectrum.phillipsA,
        windPower: SETTINGS.spectrum.windPower,
        smallWaveDamp: SETTINGS.spectrum.smallWaveDamp,
      });

      const spectrumBuffer = createAndUploadStorageBuffer(
        device,
        spectrumModes,
        'Strict Tessendorf Spectrum Buffer'
      );

      const indices = createGridIndices(FFT_SIZE);
      const indexBuffer = createIndexBuffer(device, indices, 'Strict Tessendorf Index Buffer');

      const spectrumModule = device.createShaderModule({
        label: 'Strict Tessendorf spectrum fields shader',
        code: spectrumFieldsShader,
      });

      const fftModule = device.createShaderModule({
        label: 'Strict Tessendorf FFT shader',
        code: fftShader,
      });

      const sampleModule = device.createShaderModule({
        label: 'Strict Tessendorf sample shader',
        code: sampleShader,
      });

      const renderModule = device.createShaderModule({
        label: 'Strict Tessendorf render shader',
        code: renderShader,
      });

      const spectrumPipeline = device.createComputePipeline({
        label: 'Strict Tessendorf spectrum fields pipeline',
        layout: 'auto',
        compute: {
          module: spectrumModule,
          entryPoint: 'csMain',
        },
      });

      const fftPipeline = device.createComputePipeline({
        label: 'Strict Tessendorf FFT pipeline',
        layout: 'auto',
        compute: {
          module: fftModule,
          entryPoint: 'csMain',
        },
      });

      const samplePipeline = device.createComputePipeline({
        label: 'Strict Tessendorf sample pipeline',
        layout: 'auto',
        compute: {
          module: sampleModule,
          entryPoint: 'csMain',
        },
      });

      const renderPipeline = device.createRenderPipeline({
        label: 'Strict Tessendorf render pipeline',
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
        label: 'Strict Tessendorf spectrum fields bind group',
        layout: spectrumPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fftFields.height.a } },
          { binding: 1, resource: { buffer: fftFields.gradientX.a } },
          { binding: 2, resource: { buffer: fftFields.gradientZ.a } },
          { binding: 3, resource: { buffer: fftFields.displacementX.a } },
          { binding: 4, resource: { buffer: fftFields.displacementZ.a } },
          { binding: 5, resource: { buffer: uniformBuffer } },
          { binding: 6, resource: { buffer: spectrumBuffer } },
        ],
      });

      const fftLayout = fftPipeline.getBindGroupLayout(0);
      const { rowBindGroups, columnBindGroups } = createFieldFFTBindGroups(
        device,
        fftLayout,
        fftFields
      );

      const sampleBindGroup = device.createBindGroup({
        label: 'Strict Tessendorf sample bind group',
        layout: samplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fftFields.height.a } },
          { binding: 1, resource: { buffer: fftFields.gradientX.a } },
          { binding: 2, resource: { buffer: fftFields.gradientZ.a } },
          { binding: 3, resource: { buffer: fftFields.displacementX.a } },
          { binding: 4, resource: { buffer: fftFields.displacementZ.a } },
          { binding: 5, resource: { buffer: sampleBuffer } },
          { binding: 6, resource: { buffer: uniformBuffer } },
        ],
      });

      const renderBindGroup = device.createBindGroup({
        label: 'Strict Tessendorf render bind group',
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sampleBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
      });

      const camera = new THREE.PerspectiveCamera(
        SETTINGS.camera.fov,
        1,
        SETTINGS.camera.near,
        SETTINGS.camera.far
      );

      const target = new THREE.Vector3(
        SETTINGS.camera.target.x,
        SETTINGS.camera.target.y,
        SETTINGS.camera.target.z
      );

      const viewProjection = new THREE.Matrix4();
      const uniformArray = new Float32Array(36);

      function resize() {
        if (!wrapRef.current || !canvasRef.current || !context || !device) {
          return;
        }

        const width = Math.max(1, wrapRef.current.clientWidth);
        const height = Math.max(1, wrapRef.current.clientHeight);
        const dpr = Math.min(
          window.devicePixelRatio || 1,
          SETTINGS.renderer.maxDevicePixelRatio
        );

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
          Math.sin(controls.pitch) * controls.distance +
            SETTINGS.camera.heightOffset,
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
        uniformArray[3] = SETTINGS.windSpeed;

        uniformArray[4] = SETTINGS.waveHeight;
        uniformArray[5] = SETTINGS.choppiness;
        uniformArray[6] = SETTINGS.spectrumStrength;
        uniformArray[7] = SETTINGS.foamStrength;

        uniformArray[8] = SETTINGS.bodyDetail;
        uniformArray[9] = SETTINGS.windDirection;
        uniformArray[10] = 0.0;
        uniformArray[11] = 0.0;

        uniformArray[12] = SETTINGS.sunDirection.x;
        uniformArray[13] = SETTINGS.sunDirection.y;
        uniformArray[14] = SETTINGS.sunDirection.z;
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
          label: 'Strict Tessendorf h(k,t) fields pass',
        });

        spectrumPass.setPipeline(spectrumPipeline);
        spectrumPass.setBindGroup(0, spectrumBindGroup);
        spectrumPass.dispatchWorkgroups(Math.ceil(vertexCount / 256));
        spectrumPass.end();

        for (const fieldName of FIELD_NAMES) {
          for (let stage = 0; stage < FFT_LOG_SIZE; stage += 1) {
            const rowPass = encoder.beginComputePass({
              label: `${fieldName} horizontal IFFT stage ${stage}`,
            });

            rowPass.setPipeline(fftPipeline);
            rowPass.setBindGroup(0, rowBindGroups[fieldName][stage]);
            rowPass.dispatchWorkgroups(Math.ceil(vertexCount / 2 / 256));
            rowPass.end();
          }

          for (let stage = 0; stage < FFT_LOG_SIZE; stage += 1) {
            const columnPass = encoder.beginComputePass({
              label: `${fieldName} vertical IFFT stage ${stage}`,
            });

            columnPass.setPipeline(fftPipeline);
            columnPass.setBindGroup(0, columnBindGroups[fieldName][stage]);
            columnPass.dispatchWorkgroups(Math.ceil(vertexCount / 2 / 256));
            columnPass.end();
          }
        }

        const samplePass = encoder.beginComputePass({
          label: 'Strict Tessendorf displacement + normal sample pass',
        });

        samplePass.setPipeline(samplePipeline);
        samplePass.setBindGroup(0, sampleBindGroup);
        samplePass.dispatchWorkgroups(Math.ceil(vertexCount / 256));
        samplePass.end();

        const colorView = context.getCurrentTexture().createView();
        const depthView = depthTexture.createView();

        const clear = SETTINGS.renderer.clearColor;

        const renderPass = encoder.beginRenderPass({
          label: 'Strict Tessendorf ocean render pass',
          colorAttachments: [
            {
              view: colorView,
              clearValue: {
                r: clear.r,
                g: clear.g,
                b: clear.b,
                a: clear.a,
              },
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

      setStatus('v0.19.0 Strict Tessendorf IFFT Pipeline is running.');
      setDetails(
        `${FFT_SIZE}x${FFT_SIZE} fields: height, gradient X/Z, and displacement X/Z are generated in frequency space, IFFT processed, then used for vertex height, normals, and chop.`
      );
    }

    init().catch((error) => {
      console.error(error);
      setStatus('WebGPU strict Tessendorf IFFT pipeline failed.');
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
          width: 'min(760px, calc(100vw - 56px))',
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
          OceanShader Pro / v0.19.0
        </p>

        <h1
          style={{
            margin: 0,
            fontSize: 'clamp(28px, 4vw, 48px)',
            lineHeight: 0.96,
            letterSpacing: '-0.055em',
          }}
        >
          Strict Tessendorf IFFT Pipeline
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