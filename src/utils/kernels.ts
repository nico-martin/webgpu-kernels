import { getKernel } from "@huggingface/kernels";

const USE_BUNDLED_HUB_ASSETS = import.meta.env.VITE_BUNDLE_HUB_ASSETS === "true";

export function loadKernel(opId: string) {
  if (!USE_BUNDLED_HUB_ASSETS) {
    return getKernel(`webgpu-kernels/${opId}`, { version: 1 });
  }

  const location = new URL(
    `${import.meta.env.BASE_URL}hub/kernels/webgpu-kernels/${opId}`,
    window.location.origin
  ).href;
  return getKernel(location, {
    version: 1,
    expectedOpId: opId,
    trustRemoteCode: true,
  });
}
