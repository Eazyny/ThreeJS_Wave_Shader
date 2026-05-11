export function createGridIndices(size) {
  const cells = size - 1;
  const indices = new Uint32Array(cells * cells * 6);

  let pointer = 0;

  for (let z = 0; z < cells; z += 1) {
    for (let x = 0; x < cells; x += 1) {
      const a = z * size + x;
      const b = a + 1;
      const c = a + size;
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
