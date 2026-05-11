export function createStorageBuffer(device, size, label = "Storage Buffer") {
  return device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

export function createUniformBuffer(device, size, label = "Uniform Buffer") {
  return device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function createIndexBuffer(device, indices, label = "Index Buffer") {
  const buffer = device.createBuffer({
    label,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(buffer, 0, indices);

  return buffer;
}

export function createAndUploadStorageBuffer(
  device,
  data,
  label = "Uploaded Storage Buffer",
) {
  const buffer = createStorageBuffer(device, data.byteLength, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function createFFTParamBuffer(device, stage, direction, size) {
  const values = new Uint32Array([stage, direction, size, 0]);

  const buffer = createUniformBuffer(
    device,
    values.byteLength,
    `FFT Params stage ${stage} direction ${direction}`,
  );

  device.queue.writeBuffer(buffer, 0, values);

  return buffer;
}
