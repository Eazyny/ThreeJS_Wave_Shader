import React from 'react';
import { Leva } from 'leva';
import OceanScene from './ocean/legacy/ocean-scene.jsx';
import WebGPUProbe from './ocean/webgpu/webgpu-probe.jsx';
import WebGPUFFTProbe from './ocean/webgpu/webgpu-fft-probe.jsx';

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const showWebGPUProbe = params.has('webgpu');
  const showFFTProbe = params.has('fft');

  if (showFFTProbe) {
    return <WebGPUFFTProbe />;
  }

  if (showWebGPUProbe) {
    return <WebGPUProbe />;
  }

  return (
    <main
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'hidden',
        background: '#7faecc',
        color: 'white',
        fontFamily:
          'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
      }}
    >
      <OceanScene />

      <section
        style={{
          position: 'absolute',
          left: 28,
          bottom: 28,
          width: 'min(620px, calc(100vw - 56px))',
          padding: '18px 20px',
          border: '1px solid rgba(180, 235, 255, 0.24)',
          borderRadius: 18,
          background:
            'linear-gradient(135deg, rgba(2, 18, 30, 0.58), rgba(5, 35, 50, 0.32))',
          boxShadow: '0 18px 70px rgba(0, 0, 0, 0.25)',
          backdropFilter: 'blur(14px)',
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
          OceanShader Pro
        </p>

        <h1
          style={{
            margin: 0,
            fontSize: 'clamp(28px, 4vw, 52px)',
            lineHeight: 0.96,
            letterSpacing: '-0.06em',
          }}
        >
          Ocean System
        </h1>

        <p
          style={{
            margin: '12px 0 0',
            fontSize: 14,
            lineHeight: 1.65,
            color: 'rgba(235, 250, 255, 0.86)',
          }}
        >
          Native WebGPU confirmed. Test routes:{' '}
          <span style={{ color: 'rgba(190, 245, 255, 0.95)' }}>
            ?webgpu=1
          </span>{' '}
          and{' '}
          <span style={{ color: 'rgba(190, 245, 255, 0.95)' }}>
            ?fft=1
          </span>
          .
        </p>
      </section>

      <Leva collapsed={false} />
    </main>
  );
}