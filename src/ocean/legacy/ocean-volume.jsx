import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const volumeVertexShader = `
  varying vec3 vWorldPosition;
  varying vec3 vLocalPosition;

  void main() {
    vLocalPosition = position;

    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const volumeFragmentShader = `
  precision highp float;

  uniform float uTime;
  uniform float uCameraHeight;

  uniform float uVolumeStrength;
  uniform float uUnderwaterDepth;
  uniform float uUnderwaterVisibility;
  uniform float uParticleStrength;

  uniform vec3 uSunDirection;
  uniform vec3 uDeepColor;
  uniform vec3 uMidColor;
  uniform vec3 uShallowColor;
  uniform vec3 uSunColor;

  varying vec3 vWorldPosition;
  varying vec3 vLocalPosition;

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
    float underwaterAmount =
      1.0 - smoothstep(-0.05, 0.75, uCameraHeight);

    float nearWaterline =
      1.0 - smoothstep(0.08, 1.35, abs(uCameraHeight));

    float active = max(underwaterAmount, nearWaterline * 0.32);

    if (active < 0.01) {
      discard;
    }

    vec3 viewVector = vWorldPosition - cameraPosition;
    float pathLength = length(viewVector);

    vec3 viewDir = normalize(viewVector);
    vec3 sunDir = normalize(uSunDirection);

    float visibilityScale = mix(155.0, 52.0, clamp(uUnderwaterDepth * 0.5, 0.0, 1.0));
    visibilityScale *= max(uUnderwaterVisibility, 0.2);

    float opticalDepth = pathLength / max(visibilityScale, 1.0);
    float depthFog = 1.0 - exp(-opticalDepth * uUnderwaterDepth);

    float upward = clamp(viewDir.y * 0.5 + 0.5, 0.0, 1.0);
    float forwardScatter = pow(max(dot(viewDir, sunDir), 0.0), 3.0);

    float volumeNoise = fbm(vec2(
      vWorldPosition.x * 0.018 + uTime * 0.012,
      vWorldPosition.z * 0.018 - uTime * 0.009
    ));

    float particulate = fbm(vec2(
      vWorldPosition.x * 0.11 - uTime * 0.045,
      vWorldPosition.y * 0.11 + uTime * 0.03
    ));

    vec3 shallow = mix(uShallowColor, uMidColor, 0.58);
    vec3 deep = mix(uDeepColor, vec3(0.005, 0.055, 0.075), 0.5);

    vec3 color = mix(shallow, deep, depthFog);
    color += uSunColor * forwardScatter * 0.08;
    color += uShallowColor * upward * 0.035;
    color += vec3(0.04, 0.13, 0.15) * volumeNoise * 0.08;
    color += vec3(0.12, 0.18, 0.18) * smoothstep(0.88, 0.98, particulate) * uParticleStrength * 0.035;

    float alpha = depthFog * 0.52;
    alpha += underwaterAmount * 0.08;
    alpha += nearWaterline * 0.045;
    alpha *= active;
    alpha *= uVolumeStrength;

    alpha = clamp(alpha, 0.0, 0.68);

    gl_FragColor = vec4(color, alpha);
  }
`;

export default function OceanVolume({
  volumeStrength = 0.88,
  underwaterDepth = 1.35,
  underwaterVisibility = 0.72,
  particleStrength = 0.82,
  deepColor = '#043442',
  midColor = '#086b7d',
  shallowColor = '#19aebb',
  sunColor = '#fff1d2',
}) {
  const meshRef = useRef();
  const materialRef = useRef();

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uCameraHeight: { value: 10 },

      uVolumeStrength: { value: volumeStrength },
      uUnderwaterDepth: { value: underwaterDepth },
      uUnderwaterVisibility: { value: underwaterVisibility },
      uParticleStrength: { value: particleStrength },

      uDeepColor: { value: new THREE.Color(deepColor) },
      uMidColor: { value: new THREE.Color(midColor) },
      uShallowColor: { value: new THREE.Color(shallowColor) },
      uSunColor: { value: new THREE.Color(sunColor) },

      uSunDirection: {
        value: new THREE.Vector3(-0.48, 0.58, 0.66).normalize(),
      },
    }),
    []
  );

  useFrame((state) => {
    if (!meshRef.current || !materialRef.current) return;

    meshRef.current.position.x = state.camera.position.x;
    meshRef.current.position.y = -42;
    meshRef.current.position.z = state.camera.position.z;

    const u = materialRef.current.uniforms;

    u.uTime.value = state.clock.elapsedTime;
    u.uCameraHeight.value = state.camera.position.y;

    u.uVolumeStrength.value = volumeStrength;
    u.uUnderwaterDepth.value = underwaterDepth;
    u.uUnderwaterVisibility.value = underwaterVisibility;
    u.uParticleStrength.value = particleStrength;

    u.uDeepColor.value.set(deepColor);
    u.uMidColor.value.set(midColor);
    u.uShallowColor.value.set(shallowColor);
    u.uSunColor.value.set(sunColor);
  });

  return (
    <mesh ref={meshRef} renderOrder={60} frustumCulled={false}>
      <boxGeometry args={[2200, 90, 2200, 1, 1, 1]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={volumeVertexShader}
        fragmentShader={volumeFragmentShader}
        side={THREE.BackSide}
        transparent
        depthWrite={false}
        depthTest={false}
        blending={THREE.NormalBlending}
      />
    </mesh>
  );
}