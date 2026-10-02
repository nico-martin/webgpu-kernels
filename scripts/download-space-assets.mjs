import { Buffer } from "node:buffer";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const HUB = "https://huggingface.co";
const outputRoot = resolve("dist/hub");
const kernelOps = [
  "ai.onnx.Add",
  "ai.onnx.LayerNormalization",
  "ai.onnx.MatMul",
  "ai.onnx.Softmax",
];

async function fetchFile(url, outputPath) {
  const response = await globalThis.fetch(url);
  if (!response.ok) throw new Error(`Could not download ${url} (${response.status})`);
  const data = Buffer.from(await response.arrayBuffer());
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, data);
  return { data, response };
}

async function downloadKernel(opId) {
  const repository = `webgpu-kernels/${opId}`;
  const artifactPath = "build/webgpu";
  const metadataUrl = `${HUB}/kernels/${repository}/resolve/v1/${artifactPath}/metadata.json`;
  const metadataPath = resolve(
    outputRoot,
    "kernels",
    repository,
    artifactPath,
    "metadata.json"
  );
  const { data, response } = await fetchFile(metadataUrl, metadataPath);
  const revision = response.headers.get("x-repo-commit");
  if (!revision) throw new Error(`No commit returned for ${repository}`);

  const metadata = JSON.parse(data.toString());
  await Promise.all(
    Object.keys(metadata.digest.files).map((file) =>
      fetchFile(
        `${HUB}/kernels/${repository}/resolve/${revision}/${artifactPath}/${file}`,
        resolve(outputRoot, "kernels", repository, artifactPath, file)
      )
    )
  );
}

async function downloadModelFile(repository, revision, file) {
  await fetchFile(
    `${HUB}/${repository}/resolve/${revision}/${file}`,
    resolve(outputRoot, repository, file)
  );
}

await Promise.all([
  ...kernelOps.map(downloadKernel),
  downloadModelFile(
    "google/bert_uncased_L-2_H-128_A-2",
    "30b0a37ccaaa32f332884b96992754e246e48c5f",
    "model.safetensors"
  ),
  ...["config.json", "tokenizer.json", "tokenizer_config.json"].map((file) =>
    downloadModelFile(
      "Xenova/bert-base-uncased",
      "ab680a327acc2d9c3bd279ffb1cd43454181f743",
      file
    )
  ),
]);

globalThis.console.log("Bundled Hugging Face assets in dist/hub");
