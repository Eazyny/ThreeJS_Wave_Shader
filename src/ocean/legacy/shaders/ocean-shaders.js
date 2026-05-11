export const oceanVertexShader = `
  uniform float uTime;
  uniform float uWindSpeed;
  uniform float uWindDirection;
  uniform float uFetchLength;

  uniform float uWaveHeight;
  uniform float uWaveScale;
  uniform float uWaveSpeed;
  uniform float uChoppiness;
  uniform float uSwellStrength;
  uniform float uChopStrength;

  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  varying float vSlope;
  varying float vFoam;
  varying float vCrest;
  varying float vWaterHeight;
  varying float vTroughShadow;

  const float PI = 3.141592653589793;
  const float G = 9.81;

  vec2 windDir() {
    float a = radians(uWindDirection);
    return normalize(vec2(cos(a), sin(a)));
  }

  vec2 rotate2d(vec2 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
  }

  float phase(vec2 p, vec2 dir, float wavelength, float speed, float offset) {
    dir = normalize(dir);

    float k = 2.0 * PI / wavelength;
    float omega = sqrt(G * k);

    return dot(p, dir) * k - uTime * omega * speed * uWaveSpeed + offset;
  }

  float waveHeightSample(vec2 p, vec2 dir, float wavelength, float amplitude, float speed, float offset) {
    return sin(phase(p, dir, wavelength, speed, offset)) * amplitude;
  }

  vec2 waveChopSample(vec2 p, vec2 dir, float wavelength, float amplitude, float steepness, float speed, float offset) {
    dir = normalize(dir);

    float k = 2.0 * PI / wavelength;
    float ph = phase(p, dir, wavelength, speed, offset);
    float q = clamp(steepness / (k * amplitude * 7.0 + 0.001), 0.0, 1.0);

    return dir * cos(ph) * amplitude * q;
  }

  float getHeight(vec2 p) {
    vec2 q = p * max(uWaveScale, 0.001);

    vec2 w = windDir();
    vec2 p1 = w;
    vec2 p2 = normalize(rotate2d(w, 0.38));
    vec2 p3 = normalize(rotate2d(w, -0.72));
    vec2 p4 = normalize(rotate2d(w, 1.35));

    float windEnergy = clamp(uWindSpeed / 24.0, 0.18, 1.7);
    float fetchEnergy = clamp(uFetchLength / 220.0, 0.25, 2.0);
    float swellBase = 34.0 + windEnergy * 22.0 + fetchEnergy * 10.0;

    float h = 0.0;

    h += waveHeightSample(q, p1, swellBase, 0.92 * windEnergy, 0.72, 0.0) * uSwellStrength;
    h += waveHeightSample(q, p2, swellBase * 0.68, 0.52 * windEnergy, 0.86, 1.7) * uSwellStrength;
    h += waveHeightSample(q, p3, swellBase * 0.42, 0.26 * windEnergy, 1.05, 4.1) * uSwellStrength;

    h += waveHeightSample(q, p4, 11.5, 0.13, 1.5, 0.9) * uChopStrength;
    h += waveHeightSample(q, -p2, 6.8, 0.07, 2.0, 2.8) * uChopStrength;
    h += waveHeightSample(q, -p3, 3.6, 0.038, 2.7, 5.7) * uChopStrength;

    return h * uWaveHeight;
  }

  vec2 getOffset(vec2 p) {
    vec2 q = p * max(uWaveScale, 0.001);

    vec2 w = windDir();
    vec2 p1 = w;
    vec2 p2 = normalize(rotate2d(w, 0.38));
    vec2 p3 = normalize(rotate2d(w, -0.72));
    vec2 p4 = normalize(rotate2d(w, 1.35));

    float windEnergy = clamp(uWindSpeed / 24.0, 0.18, 1.7);
    float fetchEnergy = clamp(uFetchLength / 220.0, 0.25, 2.0);
    float swellBase = 34.0 + windEnergy * 22.0 + fetchEnergy * 10.0;

    vec2 offset = vec2(0.0);

    offset += waveChopSample(q, p1, swellBase, 0.92 * windEnergy, 0.42, 0.72, 0.0) * uSwellStrength;
    offset += waveChopSample(q, p2, swellBase * 0.68, 0.52 * windEnergy, 0.32, 0.86, 1.7) * uSwellStrength;
    offset += waveChopSample(q, p3, swellBase * 0.42, 0.26 * windEnergy, 0.24, 1.05, 4.1) * uSwellStrength;

    offset += waveChopSample(q, p4, 11.5, 0.13, 0.1, 1.5, 0.9) * uChopStrength;
    offset += waveChopSample(q, -p2, 6.8, 0.07, 0.07, 2.0, 2.8) * uChopStrength;

    return offset * uChoppiness * uWaveHeight;
  }

  vec3 getSurfacePoint(vec2 p) {
    vec2 offset = getOffset(p);
    float height = getHeight(p);

    return vec3(
      p.x + offset.x,
      p.y + offset.y,
      height
    );
  }

  void main() {
    vec2 p = position.xy;

    vec3 center = getSurfacePoint(p);

    float eps = 0.25;
    vec3 px = getSurfacePoint(p + vec2(eps, 0.0));
    vec3 py = getSurfacePoint(p + vec2(0.0, eps));

    vec3 tangentX = px - center;
    vec3 tangentY = py - center;

    vec3 localNormal = normalize(cross(tangentY, tangentX));

    float slope = length(localNormal.xz);
    float crest = smoothstep(0.18, 0.92, center.z) * smoothstep(0.11, 0.55, slope);
    float foam = smoothstep(0.2, 0.64, slope) * smoothstep(0.0, 0.95, center.z + slope * 0.4);
    float troughShadow = smoothstep(0.16, 1.0, -center.z) * smoothstep(0.12, 0.68, slope);

    vSlope = slope;
    vFoam = foam;
    vCrest = crest;
    vWaterHeight = center.z;
    vTroughShadow = troughShadow;

    vec4 worldPosition = modelMatrix * vec4(center, 1.0);

    vWorldPosition = worldPosition.xyz;
    vNormal = normalize(normalMatrix * localNormal);

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

export const oceanFragmentShader = `
  precision highp float;

  uniform float uTime;
  uniform float uCameraHeight;

  uniform sampler2D uNormalMapA;
  uniform sampler2D uNormalMapB;
  uniform sampler2D uFoamNoise;

  uniform vec3 uDeepColor;
  uniform vec3 uMidColor;
  uniform vec3 uShallowColor;
  uniform vec3 uFoamColor;
  uniform vec3 uSunColor;
  uniform vec3 uFogColor;
  uniform vec3 uSkyColor;
  uniform vec3 uSunDirection;

  uniform float uWindDirection;

  uniform float uNormalStrength;
  uniform float uNormalScaleA;
  uniform float uNormalScaleB;
  uniform float uNormalSpeedA;
  uniform float uNormalSpeedB;

  uniform float uFoamStrength;
  uniform float uWindFoamStrength;
  uniform float uFoamSharpness;

  uniform float uReflectionStrength;
  uniform float uFresnelBoost;
  uniform float uSunIntensity;

  uniform float uWaterContrast;
  uniform float uBodyDetailStrength;
  uniform float uSkyFillStrength;
  uniform float uBackscatterStrength;

  uniform float uWaterIOR;
  uniform float uAbsorptionStrength;

  uniform float uSurfaceOpacity;
  uniform float uUnderwaterOpacity;
  uniform float uUnderwaterDepth;

  uniform float uFogNear;
  uniform float uFogFar;

  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  varying float vSlope;
  varying float vFoam;
  varying float vCrest;
  varying float vWaterHeight;
  varying float vTroughShadow;

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

  mat2 rotate2d(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
  }

  vec2 windDir() {
    float a = radians(uWindDirection);
    return normalize(vec2(cos(a), sin(a)));
  }

  float schlickFresnel(float cosTheta, float ior) {
    float f0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
  }

  vec3 sampleSky(vec3 dir) {
    dir = normalize(dir);

    vec3 sunDir = normalize(uSunDirection);
    float vertical = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    vec3 horizon = vec3(0.84, 0.89, 0.9);
    vec3 mid = uSkyColor;
    vec3 top = vec3(0.31, 0.52, 0.70);

    vec3 sky = mix(horizon, mid, smoothstep(0.0, 0.52, vertical));
    sky = mix(sky, top, smoothstep(0.35, 1.0, vertical));

    float sunAmount = max(dot(dir, sunDir), 0.0);
    float wideGlow = pow(sunAmount, 4.0);
    float hotGlow = pow(sunAmount, 48.0);

    sky += uSunColor * wideGlow * 0.12;
    sky += uSunColor * hotGlow * 0.85;

    return sky;
  }

  vec3 textureNormal(vec2 worldXZ, vec3 baseNormal) {
    vec2 wind = windDir();
    vec2 side = vec2(-wind.y, wind.x);

    vec2 uvA = worldXZ * uNormalScaleA + wind * uTime * uNormalSpeedA;
    vec2 uvB = rotate2d(0.65) * worldXZ * uNormalScaleB - side * uTime * uNormalSpeedB;

    vec3 nA = texture2D(uNormalMapA, uvA).xyz * 2.0 - 1.0;
    vec3 nB = texture2D(uNormalMapB, uvB).xyz * 2.0 - 1.0;

    vec2 combined = nA.xy * 0.62 + nB.xy * 0.38;

    vec3 ripple = vec3(combined.x, 0.0, combined.y) * uNormalStrength;

    return normalize(baseNormal + ripple * 0.42);
  }

  vec3 applyWaterAbsorption(vec3 baseColor, float opticalDepth) {
    vec3 sigma = vec3(2.22, 1.08, 0.30) * uAbsorptionStrength;
    vec3 transmittance = exp(-sigma * opticalDepth);
    vec3 scatter = mix(uMidColor, uShallowColor, 0.30);

    return baseColor * transmittance + scatter * (1.0 - transmittance) * 0.50;
  }

  float foamMask(vec2 worldXZ) {
    vec2 wind = windDir();
    vec2 side = vec2(-wind.y, wind.x);

    vec2 uvFoamA = worldXZ * 0.035 + wind * uTime * 0.025;
    vec2 uvFoamB = vec2(dot(worldXZ, side), dot(worldXZ, wind)) * vec2(0.08, 0.018);
    uvFoamB.y += uTime * 0.03;

    float foamTex = texture2D(uFoamNoise, uvFoamA).r;
    float streak = texture2D(uFoamNoise, uvFoamB).r;

    float brokenFoam = smoothstep(0.74, 0.92, foamTex);
    float windStreak = smoothstep(0.78, 0.95, streak);

    float mask = brokenFoam * vFoam;
    mask += windStreak * vFoam * 0.6;

    mask = pow(clamp(mask, 0.0, 1.0), max(0.55, uFoamSharpness));

    return mask;
  }

  void main() {
    vec3 N = normalize(vNormal);

    if (!gl_FrontFacing) {
      N = -N;
    }

    N = textureNormal(vWorldPosition.xz, N);

    vec3 V = normalize(cameraPosition - vWorldPosition);
    vec3 L = normalize(uSunDirection);
    vec3 R = reflect(-V, N);

    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float ndlRaw = dot(N, L);
    float ndl = clamp(ndlRaw, 0.0, 1.0);
    float grazing = 1.0 - ndv;

    float fresnel = schlickFresnel(ndv, max(uWaterIOR, 1.001));
    fresnel *= uFresnelBoost;
    fresnel = clamp(fresnel, 0.0, 1.0);

    vec2 wind = windDir();

    float longVariation = fbm(vWorldPosition.xz * 0.009 + wind * uTime * 0.006);
    float mediumVariation = fbm(vWorldPosition.xz * 0.028 - wind * uTime * 0.012);
    float normalVariation = texture2D(
      uNormalMapB,
      vWorldPosition.xz * 0.02 + wind * uTime * 0.018
    ).b;

    float waterBodyVariation =
      longVariation * 0.5 +
      mediumVariation * 0.32 +
      normalVariation * 0.18;

    vec3 baseBody = mix(uDeepColor, uMidColor, smoothstep(-1.0, 0.35, vWaterHeight));
    baseBody = mix(baseBody, uShallowColor, smoothstep(0.05, 1.05, vWaterHeight) * 0.16 + vCrest * 0.12);

    float sunSide = smoothstep(-0.22, 0.82, ndlRaw);
    float shadowSide = 1.0 - sunSide;

    float troughOcclusion = vTroughShadow * 0.32;
    float facetDarkening = shadowSide * 0.28 + troughOcclusion;
    float facetLift = sunSide * 0.13 + vCrest * 0.07;

    baseBody *= mix(
      1.0,
      mix(0.72, 1.18, waterBodyVariation),
      uBodyDetailStrength
    );

    baseBody *= 1.0 - facetDarkening;
    baseBody += uShallowColor * facetLift * 0.16;

    float opticalDepth = mix(1.1, 8.4, grazing);
    opticalDepth += vTroughShadow * 1.35;
    opticalDepth += shadowSide * 0.95;
    opticalDepth += (1.0 - waterBodyVariation) * 0.8 * uBodyDetailStrength;

    vec3 transmitted = applyWaterAbsorption(baseBody, opticalDepth);

    float forwardScatter = pow(max(dot(-V, L), 0.0), 2.0);
    forwardScatter *= 0.28 + 0.72 * vCrest;
    forwardScatter *= uBackscatterStrength;

    transmitted += uShallowColor * forwardScatter * 0.1;
    transmitted += uSkyColor * uSkyFillStrength * 0.02;

    vec3 reflectedSky = sampleSky(R);

    float reflectionMix = clamp(fresnel * uReflectionStrength, 0.02, 0.9);

    vec3 H = normalize(L + V);
    float nh = max(dot(N, H), 0.0);

    float glitterNoise = texture2D(
      uNormalMapA,
      vWorldPosition.xz * 0.18 + wind * uTime * 0.04
    ).b;

    float glitterMask = smoothstep(0.54, 0.94, glitterNoise + vSlope * 0.22);

    float specWide = pow(nh, 28.0) * 0.045;
    float specTight = pow(nh, 170.0) * 0.45;
    float specNeedle = pow(nh, 560.0) * 1.75;

    float sunSpec = (specWide + specTight + specNeedle) * glitterMask * uSunIntensity * ndl;

    vec3 color = mix(transmitted, reflectedSky, reflectionMix);
    color += uSunColor * sunSpec;

    float foam = foamMask(vWorldPosition.xz) * uFoamStrength;
    foam += foamMask(vWorldPosition.xz * 0.47 + 13.7) * uWindFoamStrength * 0.3;
    foam = clamp(foam, 0.0, 1.0);

    color = mix(color, uFoamColor, foam);

    color = mix(
      color,
      color * color * 1.15,
      clamp(uWaterContrast * 0.1, 0.0, 0.22)
    );

    float cameraUnderwater = 1.0 - smoothstep(-0.08, 0.40, uCameraHeight);
    float distanceToCamera = distance(cameraPosition, vWorldPosition);

    if (cameraUnderwater > 0.001) {
      vec3 underwaterFogColor = vec3(0.010, 0.105, 0.135);
      float underwaterDistance = smoothstep(8.0, 145.0, distanceToCamera);

      color = mix(
        color,
        underwaterFogColor,
        cameraUnderwater * underwaterDistance * uUnderwaterDepth * 0.46
      );
    }

    float distanceFog = smoothstep(uFogNear, uFogFar, distanceToCamera);
    color = mix(color, uFogColor, distanceFog * (1.0 - cameraUnderwater) * 0.12);

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(0.88));

    float alpha = mix(uSurfaceOpacity, uUnderwaterOpacity, cameraUnderwater);
    alpha = clamp(alpha, 0.08, 1.0);

    gl_FragColor = vec4(color, alpha);
  }
`;
