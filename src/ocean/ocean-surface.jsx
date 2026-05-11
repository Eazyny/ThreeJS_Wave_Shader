import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { createWaterTextures } from './utils/create-water-textures.js';
import { oceanFragmentShader, oceanVertexShader } from './shaders/ocean-shaders.js';

export default function OceanSurface(props) {
  const meshRef = useRef();
  const materialRef = useRef();

  const textures = useMemo(() => createWaterTextures(), []);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uCameraHeight: { value: 10 },

      uNormalMapA: { value: textures.normalA },
      uNormalMapB: { value: textures.normalB },
      uFoamNoise: { value: textures.foamNoise },

      uWindSpeed: { value: props.windSpeed },
      uWindDirection: { value: props.windDirection },
      uFetchLength: { value: props.fetchLength },

      uWaveHeight: { value: props.waveHeight },
      uWaveScale: { value: props.waveScale },
      uWaveSpeed: { value: props.waveSpeed },
      uChoppiness: { value: props.choppiness },
      uSwellStrength: { value: props.swellStrength },
      uChopStrength: { value: props.chopStrength },

      uNormalStrength: { value: props.normalStrength },
      uNormalScaleA: { value: props.normalScaleA },
      uNormalScaleB: { value: props.normalScaleB },
      uNormalSpeedA: { value: props.normalSpeedA },
      uNormalSpeedB: { value: props.normalSpeedB },

      uFoamStrength: { value: props.foamStrength },
      uWindFoamStrength: { value: props.windFoamStrength },
      uFoamSharpness: { value: props.foamSharpness },

      uReflectionStrength: { value: props.reflectionStrength },
      uFresnelBoost: { value: props.fresnelBoost },
      uSunIntensity: { value: props.sunIntensity },

      uWaterContrast: { value: props.waterContrast },
      uBodyDetailStrength: { value: props.bodyDetailStrength },
      uSkyFillStrength: { value: props.skyFillStrength },
      uBackscatterStrength: { value: props.backscatterStrength },

      uWaterIOR: { value: props.waterIOR },
      uAbsorptionStrength: { value: props.absorptionStrength },

      uSurfaceOpacity: { value: props.surfaceOpacity },
      uUnderwaterOpacity: { value: props.underwaterOpacity },
      uUnderwaterDepth: { value: props.underwaterDepth },

      uDeepColor: { value: new THREE.Color(props.deepColor) },
      uMidColor: { value: new THREE.Color(props.midColor) },
      uShallowColor: { value: new THREE.Color(props.shallowColor) },
      uFoamColor: { value: new THREE.Color(props.foamColor) },
      uSunColor: { value: new THREE.Color(props.sunColor) },
      uFogColor: { value: new THREE.Color(props.fogColor) },
      uSkyColor: { value: new THREE.Color(props.skyColor) },

      uSunDirection: {
        value: new THREE.Vector3(-0.48, 0.58, 0.66).normalize(),
      },

      uFogNear: { value: props.fogNear },
      uFogFar: { value: props.fogFar },
    }),
    [textures, props]
  );

  useFrame((state) => {
    if (!meshRef.current || !materialRef.current) return;

    meshRef.current.position.x = state.camera.position.x;
    meshRef.current.position.z = state.camera.position.z;

    const u = materialRef.current.uniforms;

    u.uTime.value = state.clock.elapsedTime;
    u.uCameraHeight.value = state.camera.position.y;

    u.uWindSpeed.value = props.windSpeed;
    u.uWindDirection.value = props.windDirection;
    u.uFetchLength.value = props.fetchLength;

    u.uWaveHeight.value = props.waveHeight;
    u.uWaveScale.value = props.waveScale;
    u.uWaveSpeed.value = props.waveSpeed;
    u.uChoppiness.value = props.choppiness;
    u.uSwellStrength.value = props.swellStrength;
    u.uChopStrength.value = props.chopStrength;

    u.uNormalStrength.value = props.normalStrength;
    u.uNormalScaleA.value = props.normalScaleA;
    u.uNormalScaleB.value = props.normalScaleB;
    u.uNormalSpeedA.value = props.normalSpeedA;
    u.uNormalSpeedB.value = props.normalSpeedB;

    u.uFoamStrength.value = props.foamStrength;
    u.uWindFoamStrength.value = props.windFoamStrength;
    u.uFoamSharpness.value = props.foamSharpness;

    u.uReflectionStrength.value = props.reflectionStrength;
    u.uFresnelBoost.value = props.fresnelBoost;
    u.uSunIntensity.value = props.sunIntensity;

    u.uWaterContrast.value = props.waterContrast;
    u.uBodyDetailStrength.value = props.bodyDetailStrength;
    u.uSkyFillStrength.value = props.skyFillStrength;
    u.uBackscatterStrength.value = props.backscatterStrength;

    u.uWaterIOR.value = props.waterIOR;
    u.uAbsorptionStrength.value = props.absorptionStrength;

    u.uSurfaceOpacity.value = props.surfaceOpacity;
    u.uUnderwaterOpacity.value = props.underwaterOpacity;
    u.uUnderwaterDepth.value = props.underwaterDepth;

    u.uFogNear.value = props.fogNear;
    u.uFogFar.value = props.fogFar;

    u.uDeepColor.value.set(props.deepColor);
    u.uMidColor.value.set(props.midColor);
    u.uShallowColor.value.set(props.shallowColor);
    u.uFoamColor.value.set(props.foamColor);
    u.uSunColor.value.set(props.sunColor);
    u.uFogColor.value.set(props.fogColor);
    u.uSkyColor.value.set(props.skyColor);
  });

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, 0, 0]}
      frustumCulled={false}
      renderOrder={40}
    >
      <planeGeometry args={[2200, 2200, 220, 220]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={oceanVertexShader}
        fragmentShader={oceanFragmentShader}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}