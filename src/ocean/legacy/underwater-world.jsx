import React, { useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

const SUN_DIRECTION = new THREE.Vector3(-0.48, 0.58, 0.66).normalize();

const seabedVertexShader = `
  varying vec3 vWorldPosition;
  varying float vHeightMask;

  float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
      + (c - a) * u.y * (1.0 - u.x)
      + (d - b) * u.x * u.y;
  }

  float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;

    for (int i = 0; i < 5; i++) {
      value += noise(p) * amp;
      p *= 2.0;
      amp *= 0.5;
    }

    return value;
  }

  void main() {
    vec3 pos = position;

    float duneA = fbm(pos.xy * 0.010);
    float duneB = fbm(pos.xy * 0.028 + 5.3);
    float duneC = fbm(pos.xy * 0.004 - 7.2);

    float height = (duneA - 0.5) * 3.6;
    height += (duneB - 0.5) * 1.4;
    height += (duneC - 0.5) * 6.0;

    pos.z += height;

    vHeightMask = clamp(duneA * 0.55 + duneB * 0.25 + duneC * 0.2, 0.0, 1.0);

    vec4 worldPosition = modelMatrix * vec4(pos, 1.0);
    vWorldPosition = worldPosition.xyz;

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const seabedFragmentShader = `
  precision highp float;

  varying vec3 vWorldPosition;
  varying float vHeightMask;

  uniform float uUnderwaterDepth;
  uniform float uUnderwaterVisibility;

  float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
      + (c - a) * u.y * (1.0 - u.x)
      + (d - b) * u.x * u.y;
  }

  float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;

    for (int i = 0; i < 5; i++) {
      value += noise(p) * amp;
      p *= 2.0;
      amp *= 0.5;
    }

    return value;
  }

  void main() {
    vec2 p = vWorldPosition.xz * 0.03;

    float sandNoise = fbm(p);
    float detailNoise = fbm(p * 2.5 + 4.2);
    float rockNoise = fbm(p * 5.0 - 2.0);

    vec3 sandColor = vec3(0.23, 0.24, 0.20);
    vec3 wetSandColor = vec3(0.12, 0.20, 0.185);
    vec3 algaeColor = vec3(0.045, 0.20, 0.17);
    vec3 rockColor = vec3(0.075, 0.105, 0.10);

    vec3 color = mix(sandColor, wetSandColor, smoothstep(0.25, 0.82, sandNoise));
    color = mix(color, algaeColor, smoothstep(0.66, 0.92, detailNoise) * 0.18);
    color = mix(color, rockColor, smoothstep(0.78, 0.94, rockNoise) * 0.16);

    color *= mix(0.9, 1.08, vHeightMask);

    float distanceFromOrigin = length(vWorldPosition.xz);
    float visibilityDistance = mix(280.0, 80.0, clamp(uUnderwaterDepth, 0.0, 2.0) * 0.5);
    visibilityDistance *= uUnderwaterVisibility;

    float distanceFade = smoothstep(45.0, visibilityDistance, distanceFromOrigin);
    float depthFade = smoothstep(-8.0, -42.0, vWorldPosition.y);

    vec3 depthColor = vec3(0.025, 0.105, 0.125);

    color = mix(color, depthColor, distanceFade * uUnderwaterDepth * 0.7);
    color = mix(color, depthColor, depthFade * uUnderwaterDepth * 0.42);

    gl_FragColor = vec4(color, 1.0);
  }
`;

const causticsVertexShader = `
  varying vec3 vWorldPosition;

  void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const causticsFragmentShader = `
  precision highp float;

  uniform float uTime;
  uniform float uStrength;
  uniform vec3 uSunDirection;

  varying vec3 vWorldPosition;

  float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
      + (c - a) * u.y * (1.0 - u.x)
      + (d - b) * u.x * u.y;
  }

  float ridged(float n) {
    return 1.0 - abs(n * 2.0 - 1.0);
  }

  float causticPattern(vec2 p) {
    float a = ridged(noise(p * 3.8 + vec2(uTime * 0.018, -uTime * 0.013)));
    float b = ridged(noise(p * 7.0 + vec2(-uTime * 0.025, uTime * 0.018)));
    float c = ridged(noise(p * 13.0 + vec2(uTime * 0.04, uTime * 0.02)));

    return smoothstep(0.72, 0.96, a * 0.45 + b * 0.35 + c * 0.2);
  }

  void main() {
    vec2 p = vWorldPosition.xz * 0.026;

    float pattern = causticPattern(p);

    float depthBelowSurface = max(0.0, -vWorldPosition.y);
    float shallowFade = smoothstep(32.0, 5.0, depthBelowSurface);
    float distanceFade = smoothstep(280.0, 45.0, length(vWorldPosition.xz));
    float sunAmount = smoothstep(0.08, 0.7, normalize(uSunDirection).y);

    float caustic = pattern * shallowFade * distanceFade * sunAmount * uStrength;

    vec3 color = vec3(0.72, 0.96, 0.9) * caustic;
    float alpha = caustic * 0.18;

    gl_FragColor = vec4(color, alpha);
  }
`;

const waterlineVertexShader = `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const waterlineFragmentShader = `
  precision highp float;

  uniform float uTime;
  uniform float uCameraHeight;
  uniform float uUnderwaterDepth;
  uniform float uParticleStrength;
  uniform float uWaterlineStrength;

  varying vec2 vUv;

  float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
      + (c - a) * u.y * (1.0 - u.x)
      + (d - b) * u.x * u.y;
  }

  float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;

    for (int i = 0; i < 5; i++) {
      value += noise(p) * amp;
      p *= 2.0;
      amp *= 0.5;
    }

    return value;
  }

  void main() {
    vec2 uv = vUv;

    float underwaterAmount = 1.0 - smoothstep(-0.05, 0.55, uCameraHeight);
    float nearSurfaceWindow = 1.0 - smoothstep(0.10, 0.95, abs(uCameraHeight));

    float active = max(underwaterAmount, nearSurfaceWindow * 0.52);

    if (active < 0.001) {
      discard;
    }

    float waveA = sin(uv.x * 13.0 + uTime * 0.85) * 0.018;
    float waveB = sin(uv.x * 31.0 - uTime * 0.55) * 0.008;
    float noiseWave = (fbm(vec2(uv.x * 4.0 + uTime * 0.05, uTime * 0.03)) - 0.5) * 0.018;

    float waveLine = waveA + waveB + noiseWave;
    float line = clamp(0.52 - uCameraHeight * 0.72, -0.35, 1.35);

    float underwaterMask = 1.0 - smoothstep(line - 0.04, line + 0.07, uv.y + waveLine);
    underwaterMask = max(underwaterMask, underwaterAmount * 0.8);

    float depth = smoothstep(0.0, 1.0, 1.0 - uv.y);

    float particulate = fbm(vec2(
      uv.x * 18.0 - uTime * 0.075,
      uv.y * 12.0 + uTime * 0.045
    ));

    vec3 shallowTint = vec3(0.014, 0.15, 0.19);
    vec3 deepTint = vec3(0.006, 0.06, 0.085);

    vec3 color = mix(shallowTint, deepTint, depth * 0.75 * uUnderwaterDepth);

    float particleSpark = smoothstep(0.91, 0.985, particulate);
    color += vec3(0.12, 0.16, 0.16) * particleSpark * uParticleStrength * 0.045;

    float meniscus = exp(-abs((uv.y + waveLine) - line) * 58.0);
    meniscus *= nearSurfaceWindow;
    meniscus *= uWaterlineStrength;

    vec3 meniscusColor = vec3(0.70, 0.92, 0.96);

    float volumeAlpha = underwaterMask;
    volumeAlpha *= 0.025 + depth * 0.12 * uUnderwaterDepth;
    volumeAlpha *= active;

    float meniscusAlpha = meniscus * 0.18;

    vec3 finalColor = mix(color, meniscusColor, meniscus * 0.25);

    float finalAlpha = clamp(volumeAlpha + meniscusAlpha, 0.0, 0.3);

    gl_FragColor = vec4(finalColor, finalAlpha);
  }
`;

function WaterlinePass({
  underwaterDepth,
  particleStrength,
  waterlineStrength,
}) {
  const meshRef = useRef();
  const materialRef = useRef();
  const { camera } = useThree();

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uCameraHeight: { value: 10 },
      uUnderwaterDepth: { value: underwaterDepth },
      uParticleStrength: { value: particleStrength },
      uWaterlineStrength: { value: waterlineStrength },
    }),
    []
  );

  useFrame((state) => {
    if (!meshRef.current || !materialRef.current) return;

    const distance = 0.6;
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);

    meshRef.current.position.copy(camera.position).add(direction.multiplyScalar(distance));
    meshRef.current.quaternion.copy(camera.quaternion);

    const height =
      2 * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)) * distance;
    const width = height * camera.aspect;

    meshRef.current.scale.set(width, height, 1);

    const u = materialRef.current.uniforms;

    u.uTime.value = state.clock.elapsedTime;
    u.uCameraHeight.value = camera.position.y;
    u.uUnderwaterDepth.value = underwaterDepth;
    u.uParticleStrength.value = particleStrength;
    u.uWaterlineStrength.value = waterlineStrength;
  });

  return (
    <mesh ref={meshRef} renderOrder={999}>
      <planeGeometry args={[1, 1]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={waterlineVertexShader}
        fragmentShader={waterlineFragmentShader}
        transparent
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  );
}

function Seabed({ underwaterDepth, underwaterVisibility }) {
  const materialRef = useRef();

  const uniforms = useMemo(
    () => ({
      uUnderwaterDepth: { value: underwaterDepth },
      uUnderwaterVisibility: { value: underwaterVisibility },
    }),
    []
  );

  useFrame(() => {
    if (!materialRef.current) return;

    materialRef.current.uniforms.uUnderwaterDepth.value = underwaterDepth;
    materialRef.current.uniforms.uUnderwaterVisibility.value = underwaterVisibility;
  });

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -18, -18]} receiveShadow>
      <planeGeometry args={[1600, 1600, 180, 180]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={seabedVertexShader}
        fragmentShader={seabedFragmentShader}
      />
    </mesh>
  );
}

function Seagrass() {
  const groupRef = useRef();

  const blades = useMemo(() => {
    const result = [];
    let seed = 21;

    function random() {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    }

    for (let i = 0; i < 120; i += 1) {
      result.push({
        position: [
          (random() - 0.5) * 430,
          -16.8 + random() * 1.3,
          (random() - 0.5) * 430,
        ],
        rotation: random() * Math.PI,
        scale: 0.7 + random() * 2.35,
        phase: random() * 10.0,
      });
    }

    return result;
  }, []);

  useFrame((state) => {
    if (!groupRef.current) return;

    groupRef.current.children.forEach((child, index) => {
      const blade = blades[index];
      const time = state.clock.elapsedTime;

      child.rotation.z = Math.sin(time * 0.62 + blade.phase) * 0.08;
      child.rotation.x = Math.cos(time * 0.54 + blade.phase) * 0.045;
    });
  });

  return (
    <group ref={groupRef}>
      {blades.map((blade, index) => (
        <mesh
          key={index}
          position={blade.position}
          rotation={[Math.PI, blade.rotation, 0]}
          scale={[blade.scale * 0.42, blade.scale * 2.15, blade.scale * 0.42]}
        >
          <coneGeometry args={[0.26, 4.4, 5, 1, true]} />
          <meshBasicMaterial color="#164f43" transparent opacity={0.34} />
        </mesh>
      ))}
    </group>
  );
}

function RockField() {
  const rocks = useMemo(() => {
    const result = [];
    let seed = 77;

    function random() {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    }

    for (let i = 0; i < 30; i += 1) {
      result.push({
        position: [
          (random() - 0.5) * 460,
          -16.9 + random() * 1.4,
          (random() - 0.5) * 460,
        ],
        rotation: [random() * 0.5, random() * Math.PI, random() * 0.4],
        scale: 1.2 + random() * 3.8,
      });
    }

    return result;
  }, []);

  return (
    <group>
      {rocks.map((rock, index) => (
        <mesh
          key={index}
          position={rock.position}
          rotation={rock.rotation}
          scale={[rock.scale * 1.4, rock.scale * 0.55, rock.scale]}
        >
          <dodecahedronGeometry args={[1.15, 1]} />
          <meshStandardMaterial
            color="#223d39"
            roughness={0.95}
            metalness={0.0}
            transparent
            opacity={0.72}
          />
        </mesh>
      ))}
    </group>
  );
}

function Caustics({ causticStrength }) {
  const materialRef = useRef();

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uStrength: { value: causticStrength },
      uSunDirection: {
        value: SUN_DIRECTION,
      },
    }),
    []
  );

  useFrame((state) => {
    if (!materialRef.current) return;

    materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    materialRef.current.uniforms.uStrength.value = causticStrength;
  });

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -17.3, -18]} renderOrder={4}>
      <planeGeometry args={[1600, 1600, 1, 1]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={causticsVertexShader}
        fragmentShader={causticsFragmentShader}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

function UnderwaterParticles({ particleStrength }) {
  const pointsRef = useRef();

  const geometry = useMemo(() => {
    const count = 3000;
    const positions = new Float32Array(count * 3);

    let seed = 42;

    function random() {
      seed = (seed * 16807) % 2147483647;
      return (seed - 1) / 2147483646;
    }

    for (let i = 0; i < count; i += 1) {
      positions[i * 3 + 0] = (random() - 0.5) * 720;
      positions[i * 3 + 1] = -0.65 - random() * 44;
      positions[i * 3 + 2] = (random() - 0.5) * 720;
    }

    const bufferGeometry = new THREE.BufferGeometry();
    bufferGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    return bufferGeometry;
  }, []);

  useFrame((state) => {
    if (!pointsRef.current) return;

    const time = state.clock.elapsedTime;

    pointsRef.current.rotation.y = time * 0.0028;
    pointsRef.current.position.y = Math.sin(time * 0.12) * 0.2;
    pointsRef.current.material.opacity = 0.2 * particleStrength;
  });

  return (
    <points ref={pointsRef} geometry={geometry} renderOrder={40}>
      <pointsMaterial
        color="#ddfbff"
        size={0.11}
        transparent
        opacity={0.2 * particleStrength}
        depthWrite={false}
      />
    </points>
  );
}

export default function UnderwaterWorld({
  underwaterDepth = 1.35,
  underwaterVisibility = 0.72,
  waterlineStrength = 0.62,
  particleStrength = 0.82,
  causticStrength = 0.22,
}) {
  return (
    <group>
      <Seabed
        underwaterDepth={underwaterDepth}
        underwaterVisibility={underwaterVisibility}
      />

      <RockField />
      <Seagrass />

      <Caustics causticStrength={causticStrength} />

      <UnderwaterParticles particleStrength={particleStrength} />

      <WaterlinePass
        underwaterDepth={underwaterDepth}
        particleStrength={particleStrength}
        waterlineStrength={waterlineStrength}
      />
    </group>
  );
}