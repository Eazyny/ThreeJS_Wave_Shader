import React, { useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useControls } from 'leva';
import * as THREE from 'three';
import OceanSurface from './ocean-surface.jsx';
import OceanVolume from './ocean-volume.jsx';
import UnderwaterWorld from './underwater-world.jsx';

const SUN_DIRECTION = new THREE.Vector3(-0.48, 0.58, 0.66).normalize();

function OceanEnvironment() {
  const { scene, camera } = useThree();

  const colors = useMemo(
    () => ({
      aboveBackground: new THREE.Color('#8ab4ce'),
      underwaterBackground: new THREE.Color('#062b37'),
      aboveFog: new THREE.Color('#adc8d6'),
      underwaterFog: new THREE.Color('#063744'),
    }),
    []
  );

  useFrame(() => {
    const underwaterAmount =
      1.0 - THREE.MathUtils.smoothstep(camera.position.y, -0.08, 0.82);

    if (scene.background?.isColor) {
      scene.background
        .copy(colors.aboveBackground)
        .lerp(colors.underwaterBackground, underwaterAmount);
    }

    if (scene.fog) {
      scene.fog.color
        .copy(colors.aboveFog)
        .lerp(colors.underwaterFog, underwaterAmount);

      scene.fog.near = THREE.MathUtils.lerp(320, 4, underwaterAmount);
      scene.fog.far = THREE.MathUtils.lerp(1700, 88, underwaterAmount);
    }
  });

  return null;
}

function DaySky() {
  return (
    <mesh scale={1200}>
      <sphereGeometry args={[1, 64, 64]} />
      <shaderMaterial
        side={THREE.BackSide}
        depthWrite={false}
        uniforms={{
          uTopColor: { value: new THREE.Color('#4f84b2') },
          uMidColor: { value: new THREE.Color('#8fb7d0') },
          uHorizonColor: { value: new THREE.Color('#d8ddd8') },
          uSunColor: { value: new THREE.Color('#fff2d6') },
          uSunDirection: { value: SUN_DIRECTION },
        }}
        vertexShader={`
          varying vec3 vWorldPosition;

          void main() {
            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
            vWorldPosition = worldPosition.xyz;
            gl_Position = projectionMatrix * viewMatrix * worldPosition;
          }
        `}
        fragmentShader={`
          precision highp float;

          uniform vec3 uTopColor;
          uniform vec3 uMidColor;
          uniform vec3 uHorizonColor;
          uniform vec3 uSunColor;
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
            vec3 dir = normalize(vWorldPosition);
            vec3 sunDir = normalize(uSunDirection);

            float vertical = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

            vec3 color = mix(uHorizonColor, uMidColor, smoothstep(0.0, 0.5, vertical));
            color = mix(color, uTopColor, smoothstep(0.36, 1.0, vertical));

            float sunAmount = max(dot(dir, sunDir), 0.0);
            float sunGlow = pow(sunAmount, 14.0);
            float wideGlow = pow(sunAmount, 4.0);

            color += uSunColor * wideGlow * 0.08;
            color += uSunColor * sunGlow * 0.20;

            float haze = pow(1.0 - abs(dir.y), 2.7);
            color += uHorizonColor * haze * 0.12;

            float cloudBand = smoothstep(0.12, 0.86, 1.0 - abs(dir.y - 0.24) * 4.0);
            float cloudA = fbm(vec2(dir.x * 2.0 + 0.4, dir.z * 1.1));
            float cloudB = fbm(vec2(dir.x * 5.0 - 1.2, dir.z * 2.2 + 0.7));
            float cloud = smoothstep(0.58, 0.84, cloudA * 0.65 + cloudB * 0.35) * cloudBand;

            vec3 cloudColor = mix(vec3(0.74, 0.8, 0.84), vec3(0.98, 0.92, 0.8), sunAmount);
            color = mix(color, cloudColor, cloud * 0.12);

            gl_FragColor = vec4(color, 1.0);
          }
        `}
      />
    </mesh>
  );
}

export default function OceanScene() {
  const controls = useControls('OceanShader Pro v0.17', {
    windSpeed: { value: 18.0, min: 0.0, max: 40.0, step: 0.1 },
    windDirection: { value: 25.0, min: -180.0, max: 180.0, step: 1.0 },
    fetchLength: { value: 180.0, min: 20.0, max: 500.0, step: 1.0 },

    waveHeight: { value: 0.95, min: 0.0, max: 4.0, step: 0.01 },
    waveScale: { value: 0.095, min: 0.03, max: 0.45, step: 0.001 },
    waveSpeed: { value: 0.82, min: 0.0, max: 2.5, step: 0.01 },
    choppiness: { value: 0.62, min: 0.0, max: 3.0, step: 0.01 },

    swellStrength: { value: 0.84, min: 0.0, max: 2.0, step: 0.01 },
    chopStrength: { value: 0.18, min: 0.0, max: 2.0, step: 0.01 },

    normalStrength: { value: 1.45, min: 0.0, max: 3.5, step: 0.01 },
    normalScaleA: { value: 0.052, min: 0.005, max: 0.2, step: 0.001 },
    normalScaleB: { value: 0.115, min: 0.005, max: 0.35, step: 0.001 },
    normalSpeedA: { value: 0.035, min: 0.0, max: 0.2, step: 0.001 },
    normalSpeedB: { value: 0.072, min: 0.0, max: 0.25, step: 0.001 },

    foamStrength: { value: 0.1, min: 0.0, max: 2.0, step: 0.01 },
    windFoamStrength: { value: 0.05, min: 0.0, max: 1.5, step: 0.01 },
    foamSharpness: { value: 1.65, min: 0.45, max: 3.0, step: 0.01 },

    reflectionStrength: { value: 0.82, min: 0.0, max: 1.8, step: 0.01 },
    fresnelBoost: { value: 1.18, min: 0.2, max: 3.0, step: 0.01 },
    sunIntensity: { value: 2.6, min: 0.0, max: 8.0, step: 0.01 },

    waterContrast: { value: 0.36, min: 0.0, max: 2.0, step: 0.01 },
    bodyDetailStrength: { value: 0.72, min: 0.0, max: 2.0, step: 0.01 },
    skyFillStrength: { value: 0.22, min: 0.0, max: 2.5, step: 0.01 },
    backscatterStrength: { value: 0.5, min: 0.0, max: 2.5, step: 0.01 },

    waterIOR: { value: 1.333, min: 1.0, max: 1.6, step: 0.001 },
    absorptionStrength: { value: 1.0, min: 0.0, max: 2.0, step: 0.01 },

    surfaceOpacity: { value: 1.0, min: 0.2, max: 1.0, step: 0.01 },
    underwaterOpacity: { value: 0.45, min: 0.05, max: 1.0, step: 0.01 },

    volumeStrength: { value: 0.88, min: 0.0, max: 2.0, step: 0.01 },
    underwaterDepth: { value: 1.35, min: 0.0, max: 2.0, step: 0.01 },
    underwaterVisibility: { value: 0.72, min: 0.2, max: 2.0, step: 0.01 },
    waterlineStrength: { value: 0.62, min: 0.0, max: 2.0, step: 0.01 },

    causticStrength: { value: 0.22, min: 0.0, max: 2.0, step: 0.01 },
    particleStrength: { value: 0.82, min: 0.0, max: 1.0, step: 0.01 },

    fogNear: { value: 260, min: 10, max: 900, step: 1 },
    fogFar: { value: 1450, min: 120, max: 2600, step: 1 },

    deepColor: '#043442',
    midColor: '#086b7d',
    shallowColor: '#19aebb',
    foamColor: '#eefcff',
    sunColor: '#fff1d2',
    fogColor: '#a9c3cf',
    skyColor: '#79b4d4',
  });

  return (
    <Canvas
      camera={{
        position: [0, 8.8, 52],
        fov: 46,
        near: 0.1,
        far: 2400,
      }}
      dpr={[1, 2]}
      gl={{
        antialias: true,
        powerPreference: 'high-performance',
      }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.03;
        gl.outputColorSpace = THREE.SRGBColorSpace;
      }}
    >
      <color attach="background" args={['#8ab4ce']} />
      <fog attach="fog" args={['#a9c3cf', 320, 1700]} />

      <OceanEnvironment />

      <ambientLight intensity={0.84} color="#b6e3ff" />

      <directionalLight
        position={[-75, 95, 86]}
        intensity={2.7}
        color="#fff1d2"
      />

      <DaySky />

      <UnderwaterWorld {...controls} />
      <OceanVolume {...controls} />
      <OceanSurface {...controls} />

      <OrbitControls
        enableDamping
        dampingFactor={0.055}
        minDistance={5}
        maxDistance={420}
        maxPolarAngle={Math.PI * 0.84}
        target={[0, 0.2, -55]}
      />
    </Canvas>
  );
}