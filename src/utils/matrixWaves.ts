import { getKernel } from "@huggingface/kernels";
import type { KernelGpuTensor } from "@huggingface/kernels";

export const FIELD_SIZE = 1024;
export const FIELD_CHANNELS = 128;

export interface WaveRunner {
  frame(
    time: number,
    speed: number
  ): Promise<{ values: Float32Array; workloadTime: number }>;
  destroy(): void;
}

function createBasis(): Float32Array {
  const basis = new Float32Array(FIELD_CHANNELS * FIELD_SIZE);
  const scale = 1 / Math.sqrt(FIELD_CHANNELS);
  for (let channel = 0; channel < FIELD_CHANNELS; channel += 1) {
    const frequency = 0.006 + ((channel * 17) % 47) * 0.0007;
    const phase = channel * 2.39996;
    for (let x = 0; x < FIELD_SIZE; x += 1) {
      basis[channel * FIELD_SIZE + x] =
        Math.sin(x * frequency + phase) * scale +
        Math.cos(x * frequency * 0.37 - phase) * scale * 0.45;
    }
  }
  return basis;
}

function fillSignals(target: Float32Array, time: number, speed: number) {
  const movement = time * 0.00035 * speed;
  for (let y = 0; y < FIELD_SIZE; y += 1) {
    for (let channel = 0; channel < FIELD_CHANNELS; channel += 1) {
      const frequency = 0.005 + ((channel * 29) % 53) * 0.00065;
      const phase = channel * 1.618 + movement * (0.35 + (channel % 9) * 0.08);
      target[y * FIELD_CHANNELS + channel] =
        Math.sin(y * frequency + phase) +
        Math.cos(y * frequency * 0.41 - phase * 0.72) * 0.4;
    }
  }
}

function firstGpuTensor(result: Record<string, KernelGpuTensor>): KernelGpuTensor {
  const tensor = Object.values(result)[0];
  if (!tensor) throw new Error("Kernel returned no output tensor.");
  return tensor;
}

export class GpuWaveRunner implements WaveRunner {
  private constructor(
    private readonly matmul: Awaited<ReturnType<typeof getKernel>>,
    private readonly basis: KernelGpuTensor,
    private readonly signals: Float32Array
  ) {}

  static async create(): Promise<GpuWaveRunner> {
    if (!("gpu" in navigator))
      throw new Error("WebGPU is not available in this browser.");
    const [add, matmul] = await Promise.all([
      getKernel("webgpu-kernels/ai.onnx.Add", { version: 1 }),
      getKernel("webgpu-kernels/ai.onnx.MatMul", { version: 1 }),
    ]);
    const basis = firstGpuTensor(
      await add(
        {
          a: { data: createBasis(), shape: [FIELD_CHANNELS, FIELD_SIZE] },
          b: { data: new Float32Array([0]), shape: [1] },
        },
        { output: "gpu" }
      )
    );
    return new GpuWaveRunner(
      matmul,
      basis,
      new Float32Array(FIELD_SIZE * FIELD_CHANNELS)
    );
  }

  async frame(
    time: number,
    speed: number
  ): Promise<{ values: Float32Array; workloadTime: number }> {
    fillSignals(this.signals, time, speed);
    const start = performance.now();
    const result = await this.matmul({
      a: { data: this.signals, shape: [FIELD_SIZE, FIELD_CHANNELS] },
      b: this.basis,
    });
    const tensor = Object.values(result)[0];
    if (!tensor || !(tensor.data instanceof Float32Array)) {
      throw new Error("MatMul returned an unexpected output tensor.");
    }
    return { values: tensor.data, workloadTime: performance.now() - start };
  }

  destroy() {
    this.basis.destroy();
  }
}

export class CpuWaveRunner implements WaveRunner {
  private readonly basis = createBasis();
  private readonly signals = new Float32Array(FIELD_SIZE * FIELD_CHANNELS);
  private readonly output = new Float32Array(FIELD_SIZE * FIELD_SIZE);

  async frame(
    time: number,
    speed: number
  ): Promise<{ values: Float32Array; workloadTime: number }> {
    fillSignals(this.signals, time, speed);
    const start = performance.now();
    this.output.fill(0);
    for (let y = 0; y < FIELD_SIZE; y += 1) {
      const outputOffset = y * FIELD_SIZE;
      const signalOffset = y * FIELD_CHANNELS;
      for (let channel = 0; channel < FIELD_CHANNELS; channel += 1) {
        const signal = this.signals[signalOffset + channel];
        const basisOffset = channel * FIELD_SIZE;
        for (let x = 0; x < FIELD_SIZE; x += 1) {
          this.output[outputOffset + x] += signal * this.basis[basisOffset + x];
        }
      }
    }
    return { values: this.output, workloadTime: performance.now() - start };
  }

  destroy() {}
}

export function paintWaveField(
  canvas: HTMLCanvasElement,
  values: Float32Array,
  contrast: number
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(FIELD_SIZE, FIELD_SIZE);
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.tanh(values[index] * contrast) * 0.5 + 0.5;
    const ridge = Math.max(0, 1 - Math.abs(value - 0.56) * 4.2);
    const offset = index * 4;
    image.data[offset] = 8 + value * 88 + ridge * 168;
    image.data[offset + 1] = 12 + value * value * 74 + ridge * 76;
    image.data[offset + 2] = 20 + (1 - value) * 90 + ridge * 35;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}
