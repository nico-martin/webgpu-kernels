import { getKernel } from "@huggingface/kernels";
import type { KernelGpuTensor } from "@huggingface/kernels";
import { AutoTokenizer } from "@huggingface/transformers";

const MODEL_ID = "google/bert_uncased_L-2_H-128_A-2";
const TOKENIZER_ID = "Xenova/bert-base-uncased";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/model.safetensors`;
const HIDDEN_SIZE = 128;
const HEAD_SIZE = 64;
const MAX_TOKENS = 12;

interface TensorMetadata {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

interface SafeTensors {
  buffer: ArrayBuffer;
  dataOffset: number;
  metadata: Record<string, TensorMetadata>;
}

export interface AttentionResult {
  tokens: string[];
  attention: Float32Array;
  duration: number;
  tokenCount: number;
  head: number;
}

let weightsPromise: Promise<SafeTensors> | null = null;
let tokenizerPromise: ReturnType<typeof AutoTokenizer.from_pretrained> | null = null;

async function loadWeights(onProgress: (message: string) => void): Promise<SafeTensors> {
  if (!weightsPromise) {
    weightsPromise = (async () => {
      onProgress("Downloading 17 MB pretrained BERT checkpoint");
      const response = await fetch(MODEL_URL);
      if (!response.ok)
        throw new Error(`Could not load model weights (${response.status}).`);
      const buffer = await response.arrayBuffer();
      const headerLength = Number(new DataView(buffer).getBigUint64(0, true));
      const metadata = JSON.parse(
        new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength))
      ) as Record<string, TensorMetadata>;
      return { buffer, dataOffset: 8 + headerLength, metadata };
    })();
  }
  return weightsPromise;
}

function tensor(weights: SafeTensors, name: string): Float32Array {
  const info = weights.metadata[name];
  if (!info || info.dtype !== "F32") throw new Error(`Missing float32 weight: ${name}`);
  const [start, end] = info.data_offsets;
  return new Float32Array(weights.buffer, weights.dataOffset + start, (end - start) / 4);
}

function transposeRows(
  source: Float32Array,
  sourceColumns: number,
  rows: number,
  rowOffset = 0
): Float32Array {
  const output = new Float32Array(sourceColumns * rows);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < sourceColumns; column += 1) {
      output[column * rows + row] = source[(row + rowOffset) * sourceColumns + column];
    }
  }
  return output;
}

function transpose(source: Float32Array, rows: number, columns: number): Float32Array {
  const output = new Float32Array(source.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = source[row * columns + column];
    }
  }
  return output;
}

function firstGpu(result: Record<string, KernelGpuTensor>): KernelGpuTensor {
  const value = Object.values(result)[0];
  if (!value) throw new Error("Kernel returned no GPU tensor.");
  return value;
}

function cpuData(result: Awaited<ReturnType<Awaited<ReturnType<typeof getKernel>>>>) {
  const value = Object.values(result)[0];
  if (!value || !("data" in value) || !(value.data instanceof Float32Array)) {
    throw new Error("Kernel returned no float32 tensor.");
  }
  return value.data;
}

export async function runAttention(
  sentence: string,
  onProgress: (message: string) => void,
  head = 1
): Promise<AttentionResult> {
  if (!("gpu" in navigator)) throw new Error("WebGPU is required for this demo.");
  if (head !== 0 && head !== 1) throw new Error(`Unsupported attention head: ${head}.`);
  const [weights, tokenizer, kernels] = await Promise.all([
    loadWeights(onProgress),
    (tokenizerPromise ??= AutoTokenizer.from_pretrained(TOKENIZER_ID)),
    Promise.all([
      getKernel("webgpu-kernels/ai.onnx.MatMul", { version: 1 }),
      getKernel("webgpu-kernels/ai.onnx.Softmax", { version: 1 }),
      getKernel("webgpu-kernels/ai.onnx.LayerNormalization", { version: 1 }),
      getKernel("webgpu-kernels/ai.onnx.Add", { version: 1 }),
    ]),
  ]);
  const [matmul, softmax, layerNorm, add] = kernels;
  onProgress("Tokenizing the sentence");
  const encoded = tokenizer(sentence, { truncation: true, max_length: MAX_TOKENS });
  const inputIds = Array.from(encoded.input_ids.data, Number).slice(0, MAX_TOKENS);
  const tokens = inputIds.map((id) =>
    tokenizer.decode([id], { skip_special_tokens: false })
  );
  const tokenCount = inputIds.length;
  const hidden = new Float32Array(tokenCount * HIDDEN_SIZE);
  const words = tensor(weights, "bert.embeddings.word_embeddings.weight");
  const positions = tensor(weights, "bert.embeddings.position_embeddings.weight");
  const tokenTypes = tensor(weights, "bert.embeddings.token_type_embeddings.weight");
  for (let token = 0; token < tokenCount; token += 1) {
    for (let feature = 0; feature < HIDDEN_SIZE; feature += 1) {
      hidden[token * HIDDEN_SIZE + feature] =
        words[inputIds[token] * HIDDEN_SIZE + feature] +
        positions[token * HIDDEN_SIZE + feature] +
        tokenTypes[feature];
    }
  }

  const headOffset = head * HEAD_SIZE;
  const projection = (name: "query" | "key" | "value") =>
    transposeRows(
      tensor(weights, `bert.encoder.layer.0.attention.self.${name}.weight`),
      HIDDEN_SIZE,
      HEAD_SIZE,
      headOffset
    );
  const qWeight = projection("query");
  for (let index = 0; index < qWeight.length; index += 1) {
    qWeight[index] /= Math.sqrt(HEAD_SIZE);
  }
  const kWeight = projection("key");
  const vWeight = projection("value");
  const hiddenInput = { data: hidden, shape: [tokenCount, HIDDEN_SIZE] };
  const gpu = { output: "gpu" } as const;

  onProgress("Running attention with WebGPU kernels");
  const start = performance.now();
  const normalizedInput = firstGpu(
    await layerNorm(
      {
        x: hiddenInput,
        scale: {
          data: tensor(weights, "bert.embeddings.LayerNorm.weight"),
          shape: [HIDDEN_SIZE],
        },
        b: {
          data: tensor(weights, "bert.embeddings.LayerNorm.bias"),
          shape: [HIDDEN_SIZE],
        },
      },
      { attrs: { axis: -1, epsilon: 1e-12 }, output: "gpu" }
    )
  );
  const [qRawResult, kRawResult, vRawResult] = await Promise.all([
    matmul(
      { a: normalizedInput, b: { data: qWeight, shape: [HIDDEN_SIZE, HEAD_SIZE] } },
      gpu
    ),
    matmul(
      { a: normalizedInput, b: { data: kWeight, shape: [HIDDEN_SIZE, HEAD_SIZE] } },
      gpu
    ),
    matmul(
      { a: normalizedInput, b: { data: vWeight, shape: [HIDDEN_SIZE, HEAD_SIZE] } },
      gpu
    ),
  ]);
  const qRaw = firstGpu(qRawResult);
  const kRaw = firstGpu(kRawResult);
  const vRaw = firstGpu(vRawResult);
  const qBias = tensor(weights, "bert.encoder.layer.0.attention.self.query.bias").slice(
    headOffset,
    headOffset + HEAD_SIZE
  );
  for (let index = 0; index < qBias.length; index += 1) {
    qBias[index] /= Math.sqrt(HEAD_SIZE);
  }
  const [qResult, kResult, vResult] = await Promise.all([
    add({ a: qRaw, b: { data: qBias, shape: [HEAD_SIZE] } }, gpu),
    add({
      a: kRaw,
      b: {
        data: tensor(weights, "bert.encoder.layer.0.attention.self.key.bias").slice(
          headOffset,
          headOffset + HEAD_SIZE
        ),
        shape: [HEAD_SIZE],
      },
    }),
    add(
      {
        a: vRaw,
        b: {
          data: tensor(weights, "bert.encoder.layer.0.attention.self.value.bias").slice(
            headOffset,
            headOffset + HEAD_SIZE
          ),
          shape: [HEAD_SIZE],
        },
      },
      gpu
    ),
  ]);
  qRaw.destroy();
  kRaw.destroy();
  vRaw.destroy();
  const query = firstGpu(qResult);
  const value = firstGpu(vResult);
  const key = cpuData(kResult);
  const keyTranspose = transpose(key, tokenCount, HEAD_SIZE);
  const scores = firstGpu(
    await matmul(
      { a: query, b: { data: keyTranspose, shape: [HEAD_SIZE, tokenCount] } },
      gpu
    )
  );
  query.destroy();
  const probabilities = firstGpu(
    await softmax({ x: scores }, { attrs: { axis: -1 }, output: "gpu" })
  );
  scores.destroy();
  const context = firstGpu(await matmul({ a: probabilities, b: value }, gpu));
  value.destroy();

  const outputWeight = tensor(
    weights,
    "bert.encoder.layer.0.attention.output.dense.weight"
  );
  const headOutputWeight = new Float32Array(HEAD_SIZE * HIDDEN_SIZE);
  for (let input = 0; input < HEAD_SIZE; input += 1) {
    for (let output = 0; output < HIDDEN_SIZE; output += 1) {
      headOutputWeight[input * HIDDEN_SIZE + output] =
        outputWeight[output * HIDDEN_SIZE + headOffset + input];
    }
  }
  const projected = firstGpu(
    await matmul(
      { a: context, b: { data: headOutputWeight, shape: [HEAD_SIZE, HIDDEN_SIZE] } },
      gpu
    )
  );
  context.destroy();
  const projectedWithBias = firstGpu(
    await add(
      {
        a: projected,
        b: {
          data: tensor(weights, "bert.encoder.layer.0.attention.output.dense.bias"),
          shape: [HIDDEN_SIZE],
        },
      },
      gpu
    )
  );
  projected.destroy();
  const residual = firstGpu(await add({ a: projectedWithBias, b: normalizedInput }, gpu));
  projectedWithBias.destroy();
  normalizedInput.destroy();
  const normalized = await layerNorm(
    {
      x: residual,
      scale: {
        data: tensor(weights, "bert.encoder.layer.0.attention.output.LayerNorm.weight"),
        shape: [HIDDEN_SIZE],
      },
      b: {
        data: tensor(weights, "bert.encoder.layer.0.attention.output.LayerNorm.bias"),
        shape: [HIDDEN_SIZE],
      },
    },
    { attrs: { axis: -1, epsilon: 1e-12 } }
  );
  cpuData(normalized);
  residual.destroy();

  const attentionResult = await add({
    a: probabilities,
    b: { data: new Float32Array([0]), shape: [1] },
  });
  probabilities.destroy();
  return {
    tokens,
    attention: cpuData(attentionResult),
    duration: performance.now() - start,
    tokenCount,
    head,
  };
}
