import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three/webgpu';

export default function WebGPUProbe() {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);

  const [status, setStatus] = useState('Initializing WebGPU renderer...');
  const [nativeWebGPU, setNativeWebGPU] = useState(false);

  useEffect(() => {
    let renderer = null;
    let resizeObserver = null;
    let disposed = false;

    const hasNativeWebGPU =
      typeof navigator !== 'undefined' && Boolean(navigator.gpu);

    setNativeWebGPU(hasNativeWebGPU);

    async function init() {
      if (!canvasRef.current || !wrapRef.current) return;

      try {
        renderer = new THREE.WebGPURenderer({
          canvas: canvasRef.current,
          antialias: true,
          alpha: false,
        });

        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.05;

        await renderer.init();

        if (disposed) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#07131f');

        const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
        camera.position.set(0, 1.4, 6);

        const ambient = new THREE.AmbientLight('#8fcfff', 0.65);
        scene.add(ambient);

        const sun = new THREE.DirectionalLight('#fff1d2', 3.2);
        sun.position.set(-4, 7, 5);
        scene.add(sun);

        const group = new THREE.Group();
        scene.add(group);

        const cubeGeometry = new THREE.BoxGeometry(1.55, 1.55, 1.55, 8, 8, 8);
        const cubeMaterial = new THREE.MeshStandardMaterial({
          color: '#19aebb',
          roughness: 0.28,
          metalness: 0.02,
        });

        const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
        cube.position.y = 0.3;
        group.add(cube);

        const floorGeometry = new THREE.PlaneGeometry(18, 18, 1, 1);
        const floorMaterial = new THREE.MeshStandardMaterial({
          color: '#062f3d',
          roughness: 0.8,
          metalness: 0.0,
        });

        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -1.1;
        scene.add(floor);

        function resize() {
          if (!wrapRef.current || !renderer) return;

          const width = wrapRef.current.clientWidth;
          const height = wrapRef.current.clientHeight;

          renderer.setSize(width, height, false);

          camera.aspect = width / Math.max(height, 1);
          camera.updateProjectionMatrix();
        }

        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(wrapRef.current);
        resize();

        const clock = new THREE.Clock();

        renderer.setAnimationLoop(() => {
          const elapsed = clock.getElapsedTime();

          cube.rotation.x = elapsed * 0.42;
          cube.rotation.y = elapsed * 0.68;

          group.position.y = Math.sin(elapsed * 1.2) * 0.08;

          renderer.render(scene, camera);
        });

        setStatus(
          hasNativeWebGPU
            ? 'WebGPU renderer initialized. Native WebGPU detected.'
            : 'Renderer initialized, but native WebGPU was not detected. This may be WebGL2 fallback.'
        );
      } catch (error) {
        console.error(error);
        setStatus(`WebGPU probe failed: ${error.message}`);
      }
    }

    init();

    return () => {
      disposed = true;

      if (resizeObserver) {
        resizeObserver.disconnect();
      }

      if (renderer) {
        renderer.setAnimationLoop(null);
        renderer.dispose();
      }
    };
  }, []);

  return (
    <section
      ref={wrapRef}
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'hidden',
        background: '#07131f',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 28,
          bottom: 28,
          width: 'min(560px, calc(100vw - 56px))',
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
          OceanShader Pro / v0.18 foundation
        </p>

        <h1
          style={{
            margin: 0,
            fontSize: 'clamp(28px, 4vw, 48px)',
            lineHeight: 0.96,
            letterSpacing: '-0.055em',
          }}
        >
          WebGPU Probe
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
            margin: '10px 0 0',
            fontSize: 13,
            lineHeight: 1.55,
            color: nativeWebGPU
              ? 'rgba(150, 255, 210, 0.9)'
              : 'rgba(255, 215, 150, 0.92)',
          }}
        >
          Native WebGPU detected: {nativeWebGPU ? 'YES' : 'NO / FALLBACK'}
        </p>
      </div>
    </section>
  );
}