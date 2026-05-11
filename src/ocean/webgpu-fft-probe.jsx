import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

const GRID_SIZE = 160;
const SPECTRUM_SIZE = 24;
const SPECTRUM_MODE_COUNT = SPECTRUM_SIZE * SPECTRUM_SIZE;
const OCEAN_SIZE = 320;

const computeShader = /* wgsl */ `
struct Sample {
  position: vec4f,
  normal: vec4f,
};

struct SpectrumMode {
  data0: vec4f,
  data1: vec4f,
};

struct Params {
  time: f32,
  gridSize: f32,
  oceanSize: f32,
  windSpeed: f32,

  waveScale: f32,
  waveHeight: f32,
  choppiness: f32,
  windDirection: f32,

  spectrumStrength: f32,
  swellBias: f32,
  foamStrength: f32,
  bodyDetail: f32,

  sunDir: vec4f,
  cameraPos: vec4f,
  viewProjection: mat4x4f,
};

@group(0) @binding(0) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> modes: array<SpectrumMode>;

const MODE_COUNT: u32 = ${SPECTRUM_MODE_COUNT}u;

fn spectrum_height_chop(worldXZ: vec2f) -> vec4f {
  var height = 0.0;
  var chop = vec2f(0.0);

  let scaledXZ = worldXZ * max(params.waveScale, 0.001);

  for (var i: u32 = 0u; i < MODE_COUNT; i = i + 1u) {
    let mode = modes[i];

    let k = mode.data0.xy;
    let amp = mode.data0.z;
    let phase0 = mode.data0.w;
    let omega = mode.data1.x;
    let directionWeight = mode.data1.y;

    let kLen = max(length(k), 0.0001);
    let kDir = k / kLen;

    let phase = dot(k, scaledXZ) + omega * params.time + phase0;

    let c = cos(phase);
    let s = sin(phase);

    let weightedAmp = amp * params.waveHeight * params.spectrumStrength;

    height += c * weightedAmp;

    let chopPower = params.choppiness * mix(0.45, 1.15, directionWeight);
    chop += kDir * s * weightedAmp * chopPower;
  }

  return vec4f(height, chop.x, chop.y, 0.0);
}

@compute @workgroup_size(256)
fn csMain(@builtin(global_invocation_id) globalId: vec3u) {
  let id = globalId.x;

  let gridSize = u32(params.gridSize);
  let total = gridSize * gridSize;

  if (id >= total) {
    return;
  }

  let ix = id % gridSize;
  let iz = id / gridSize;

  let gx = f32(ix) / f32(gridSize - 1u) - 0.5;
  let gz = f32(iz) / f32(gridSize - 1u) - 0.5;

  let base = vec2f(gx * params.oceanSize, gz * params.oceanSize);

  let result = spectrum_height_chop(base);

  let height = result.x;
  let chopX = result.y;
  let chopZ = result.z;

  let eps = 0.65;

  let hL = spectrum_height_chop(base - vec2f(eps, 0.0)).x;
  let hR = spectrum_height_chop(base + vec2f(eps, 0.0)).x;
  let hD = spectrum_height_chop(base - vec2f(0.0, eps)).x;
  let hU = spectrum_height_chop(base + vec2f(0.0, eps)).x;

  let normal = normalize(vec3f(
    -(hR - hL),
    2.0 * eps,
    -(hU - hD)
  ));

  samples[id].position = vec4f(
    base.x + chopX,
    height,
    base.y + chopZ,
    height
  );

  samples[id].normal = vec4f(normal, 0.0);
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

  waveScale: f32,
  waveHeight: f32,
  choppiness: f32,
  windDirection: f32,

  spectrumStrength: f32,
  swellBias: f32,
  foamStrength: f32,
  bodyDetail: f32,

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

  let windAngle = radians(params.windDirection);
  let wind = normalize(vec2f(cos(windAngle), sin(windAngle)));

  let longVariation = fbm(input.worldPosition.xz * 0.012 + wind * params.time * 0.008);
  let mediumVariation = fbm(input.worldPosition.xz * 0.045 - wind * params.time * 0.022);
  let detailVariation = fbm(input.worldPosition.xz * 0.12 + vec2f(params.time * 0.035, -params.time * 0.018));

  let variation =
    longVariation * 0.52 +
    mediumVariation * 0.33 +
    detailVariation * 0.15;

  let deepColor = vec3f(0.008, 0.13, 0.18);
  let midColor = vec3f(0.018, 0.38, 0.48);
  let shallowColor = vec3f(0.075, 0.75, 0.82);

  var body = mix(deepColor, midColor, smoothstep(-1.25, 0.45, height));
  body = mix(body, shallowColor, smoothstep(0.15, 1.45, height) * 0.18);

  let sunSide = smoothstep(-0.28, 0.82, ndlRaw);
  let shadowSide = 1.0 - sunSide;

  let trough = smoothstep(0.12, 1.15, -height);
  let crest = smoothstep(0.15, 1.25, height);
  let steepCrest = smoothstep(0.16, 0.52, slope) * crest;

  let bodyDetail = params.bodyDetail;

  body *= mix(1.0, mix(0.68, 1.24, variation), bodyDetail);
  body *= 1.0 - shadowSide * 0.28;
  body *= 1.0 - trough * 0.26;
  body += shallowColor * crest * 0.08;
  body += shallowColor * sunSide * 0.045;

  let absorption = vec3f(2.18, 1.02, 0.28);

  var opticalDepth = mix(1.1, 8.6, grazing);
  opticalDepth += trough * 1.3;
  opticalDepth += shadowSide * 0.85;
  opticalDepth += (1.0 - variation) * 0.85 * bodyDetail;

  let transmittance = exp(-absorption * opticalDepth);
  let scatter = mix(midColor, shallowColor, 0.33);
  let transmitted = body * transmittance + scatter * (1.0 - transmittance) * 0.52;

  let R = reflect(-V, N);
  let reflectedSky = sample_sky(R);

  let fresnel = clamp(schlick_fresnel(ndv, 1.333) * 1.25, 0.0, 1.0);
  let reflectionAmount = clamp(fresnel * 0.88, 0.02, 0.92);

  let H = normalize(L + V);
  let nh = max(dot(N, H), 0.0);

  let glitterNoise = fbm(input.worldPosition.xz * 0.22 + wind * params.time * 0.052);
  let glitterMask = smoothstep(0.56, 0.94, glitterNoise + slope * 0.22);

  let spec =
    pow(nh, 32.0) * 0.055 +
    pow(nh, 170.0) * 0.50 +
    pow(nh, 620.0) * 1.85;

  let sunSpec = spec * glitterMask * ndl * 2.75;

  var color = mix(transmitted, reflectedSky, reflectionAmount);
  color += vec3f(1.0, 0.90, 0.70) * sunSpec;

  let foamNoise = fbm(input.worldPosition.xz * 0.105 + wind * params.time * 0.03);
  let foamSource = smoothstep(0.32, 1.18, steepCrest + trough * 0.25 + grazing * 0.16);
  let foam = smoothstep(0.87, 0.98, foamNoise) * foamSource * params.foamStrength;

  color = mix(color, vec3f(0.92, 0.98, 1.0), clamp(foam, 0.0, 1.0));

  let distanceToCamera = length(params.cameraPos.xyz - input.worldPosition);
  let fogAmount = smoothstep(310.0, 1050.0, distanceToCamera);
  color = mix(color, vec3f(0.70, 0.82, 0.88), fogAmount * 0.16);

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
  spectrumSize,
  oceanSize,
  windSpeed,
  windDirection,
  targetStd = 1.0,
}) {
  const random = mulberry32(1337);
  const modes = new Float32Array(spectrumSize * spectrumSize * 8);

  const g = 9.81;
  const windAngle = THREE.MathUtils.degToRad(windDirection);
  const wind = new THREE.Vector2(Math.cos(windAngle), Math.sin(windAngle));

  const L = Math.max((windSpeed * windSpeed) / g, 0.001);
  const smallWaveDamp = 0.018;

  let pointer = 0;
  let amplitudeEnergy = 0;

  for (let y = 0; y < spectrumSize; y += 1) {
    for (let x = 0; x < spectrumSize; x += 1) {
      const sx = x - spectrumSize / 2;
      const sy = y - spectrumSize / 2;

      const kx = (2 * Math.PI * sx) / oceanSize;
      const kz = (2 * Math.PI * sy) / oceanSize;

      const kLength = Math.sqrt(kx * kx + kz * kz);

      let amplitude = 0;
      let directionWeight = 0;

      if (kLength > 0.00001) {
        const kDir = new THREE.Vector2(kx / kLength, kz / kLength);
        const alignment = kDir.dot(wind);

        const forward = Math.max(alignment, 0);
        const backward = Math.max(-alignment, 0);

        directionWeight = Math.pow(forward, 6) + Math.pow(backward, 2) * 0.04;

        const phillips =
          (Math.exp(-1 / Math.pow(kLength * L, 2)) /
            Math.pow(kLength, 4)) *
          directionWeight *
          Math.exp(-kLength * kLength * smallWaveDamp * smallWaveDamp);

        const randomEnergy = Math.abs(gaussianRandom(random)) * 0.35 + 0.82;

        amplitude = Math.sqrt(Math.max(phillips, 0)) * randomEnergy;
      }

      const phase = random() * Math.PI * 2;
      const omega = Math.sqrt(g * kLength);

      modes[pointer + 0] = kx;
      modes[pointer + 1] = kz;
      modes[pointer + 2] = amplitude;
      modes[pointer + 3] = phase;

      modes[pointer + 4] = omega;
      modes[pointer + 5] = directionWeight;
      modes[pointer + 6] = 0;
      modes[pointer + 7] = 0;

      amplitudeEnergy += amplitude * amplitude;

      pointer += 8;
    }
  }

  const currentStd = Math.sqrt(Math.max(amplitudeEnergy * 0.5, 0.000001));
  const normalizeFactor = targetStd / currentStd;

  for (let i = 0; i < modes.length; i += 8) {
    modes[i + 2] *= normalizeFactor;
  }

  return modes;
}

function createGridIndices(gridSize) {
  const cells = gridSize - 1;
  const indices = new Uint32Array(cells * cells * 6);

  let pointer = 0;

  for (let z = 0; z < cells; z += 1) {
    for (let x = 0; x < cells; x += 1) {
      const a = z * gridSize + x;
      const b = a + 1;
      const c = a + gridSize;
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

export default function WebGPUFFTProbe() {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const frameRef = useRef(null);

  const controlRef = useRef({
    yaw: 0.02,
    pitch: -0.12,
    distance: 82,
    dragging: false,
    lastX: 0,
    lastY: 0,
  });

  const [status, setStatus] = useState(
    'Initializing WebGPU spectrum foundation...'
  );

  const [details, setDetails] = useState(
    'Preparing spectrum buffer + compute pipeline.'
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

      const vertexCount = GRID_SIZE * GRID_SIZE;
      const sampleBufferSize = vertexCount * 8 * 4;

      const sampleBuffer = device.createBuffer({
        size: sampleBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const uniformBuffer = device.createBuffer({
        size: 36 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const spectrumModes = createSpectrumModes({
        spectrumSize: SPECTRUM_SIZE,
        oceanSize: OCEAN_SIZE,
        windSpeed: 18,
        windDirection: 25,
        targetStd: 0.92,
      });

      const spectrumBuffer = device.createBuffer({
        size: spectrumModes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      device.queue.writeBuffer(spectrumBuffer, 0, spectrumModes);

      const indices = createGridIndices(GRID_SIZE);

      const indexBuffer = device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });

      device.queue.writeBuffer(indexBuffer, 0, indices);

      const computeModule = device.createShaderModule({
        label: 'Ocean spectrum compute shader',
        code: computeShader,
      });

      const renderModule = device.createShaderModule({
        label: 'Ocean spectrum render shader',
        code: renderShader,
      });

      const computePipeline = device.createComputePipeline({
        label: 'Ocean spectrum compute pipeline',
        layout: 'auto',
        compute: {
          module: computeModule,
          entryPoint: 'csMain',
        },
      });

      const renderPipeline = device.createRenderPipeline({
        label: 'Ocean spectrum render pipeline',
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

      const computeBindGroup = device.createBindGroup({
        label: 'Ocean spectrum compute bind group',
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sampleBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
          { binding: 2, resource: { buffer: spectrumBuffer } },
        ],
      });

      const renderBindGroup = device.createBindGroup({
        label: 'Ocean spectrum render bind group',
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sampleBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
      });

      const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 2400);
      const target = new THREE.Vector3(0, 0.2, -40);
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

        uniformArray[0] = timeSeconds;
        uniformArray[1] = GRID_SIZE;
        uniformArray[2] = OCEAN_SIZE;
        uniformArray[3] = 18.0;

        uniformArray[4] = 1.0;
        uniformArray[5] = 1.15;
        uniformArray[6] = 0.92;
        uniformArray[7] = 25.0;

        uniformArray[8] = 1.0;
        uniformArray[9] = 1.0;
        uniformArray[10] = 0.18;
        uniformArray[11] = 0.85;

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

        const computePass = encoder.beginComputePass({
          label: 'Ocean spectrum compute pass',
        });

        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(vertexCount / 256));
        computePass.end();

        const colorView = context.getCurrentTexture().createView();
        const depthView = depthTexture.createView();

        const renderPass = encoder.beginRenderPass({
          label: 'Ocean spectrum render pass',
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

      setStatus('v0.18-B Spectrum Foundation is running.');
      setDetails(
        `${GRID_SIZE}x${GRID_SIZE} mesh using ${SPECTRUM_MODE_COUNT} uploaded spectrum modes. Drag to orbit, wheel to zoom.`
      );
    }

    init().catch((error) => {
      console.error(error);
      setStatus('WebGPU spectrum foundation failed.');
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
      controls.distance = Math.max(24, Math.min(220, controls.distance));
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
          width: 'min(650px, calc(100vw - 56px))',
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
          OceanShader Pro / v0.18-B
        </p>

        <h1
          style={{
            margin: 0,
            fontSize: 'clamp(28px, 4vw, 48px)',
            lineHeight: 0.96,
            letterSpacing: '-0.055em',
          }}
        >
          Spectrum Foundation
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