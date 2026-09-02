import { ArrowDown, ArrowUpRight } from "lucide-react";
import AttentionDemo from "./components/AttentionDemo";
import CodeBlock from "./components/CodeBlock";
import KernelBalloon from "./components/KernelBalloon";
import MatrixWaves from "./components/MatrixWaves";
import UnderTheHoodModal from "./components/UnderTheHoodModal";
import { Button } from "./theme";

const QUICK_START = `import { getKernel } from "@huggingface/kernels";

const add = await getKernel(
  "webgpu-kernels/ai.onnx.Add",
  { version: 1 } 
);

const { c } = await add({
  a: { data: new Float32Array([1, 2, 3]), shape: [3] },
  b: { data: new Float32Array([10]), shape: [1] },
});

console.log(c.data); // [11, 12, 13]`;

export default function App() {
  return (
    <main className="overflow-hidden bg-cream text-ink">
      <nav className="fixed inset-x-0 top-0 z-50 border-b border-white/10 bg-ink/85 text-white backdrop-blur-xl">
        <div className="page-wrap flex h-16 items-center justify-between">
          <a
            href="#top"
            className="flex items-center gap-2.5 font-semibold tracking-tight"
          >
            <img
              src="/huggingface-logo.svg"
              alt="Hugging Face"
              className="h-8 w-auto"
            />
            <span className="hidden sm:inline">WebGPU Kernels</span>
          </a>
          <div className="flex items-center gap-5 font-mono text-[11px] uppercase tracking-[0.12em] text-white/65">
            <a className="hidden hover:text-white sm:inline" href="#how-it-works">
              How it works
            </a>
            <a className="hover:text-white" href="#demos">
              Demos
            </a>
            <a
              className="flex items-center gap-1.5 hover:text-white"
              href="https://www.npmjs.com/package/@huggingface/kernels"
              target="_blank"
              rel="noreferrer"
            >
              npm <ArrowUpRight size={12} />
            </a>
          </div>
        </div>
      </nav>

      <header id="top" className="relative min-h-215 bg-ink pt-16 text-white">
        <div className="hero-grid absolute inset-0 opacity-30" />
        <div className="page-wrap relative grid min-h-198.5 items-center gap-16 py-24 lg:grid-cols-[1.18fr_0.82fr]">
          <div>
            <div className="mb-8 flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.18em] text-mint">
              <span className="size-2 animate-pulse rounded-full bg-mint" />
              207 kernels · WebGPU · Apache 2.0
            </div>
            <h1 className="max-w-225 text-[clamp(4rem,9vw,8.6rem)] font-semibold leading-[0.82] tracking-[-0.075em]">
              GPU math,
              <br />
              <span className="font-serif font-normal italic text-coral">
                from the Hub.
              </span>
            </h1>
            <p className="mt-10 max-w-xl text-lg leading-8 text-white/58">
              Load production-ready WebGPU operations with one JavaScript call. No shader
              toolchain, no server, no framework lock-in.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <a href="#demos">
                <Button>
                  Run the demos <ArrowDown size={15} />
                </Button>
              </a>
              <a href="#how-it-works">
                <Button variant="ghost">
                  How it works <ArrowDown size={15} />
                </Button>
              </a>
            </div>
          </div>

          <KernelBalloon />
        </div>
      </header>

      <section id="how-it-works" className="page-wrap py-24 sm:py-32">
        <div className="grid gap-16 lg:grid-cols-2 lg:gap-24">
          <div>
            <p className="eyebrow">The missing low-level layer</p>
            <h2 className="section-title mt-5">
              A kernel is one fast, focused operation.
            </h2>
            <p className="body-copy mt-7">
              Every AI model eventually becomes a sequence of additions, matrix
              multiplications, convolutions, and normalizations. WebGPU kernels are the
              tiny programs that execute those operations on your GPU.
            </p>
            <UnderTheHoodModal className="mt-7" />
          </div>
          <div>
            <div className="mb-4 flex items-center justify-between">
              <span className="eyebrow">Install</span>
              <code className="rounded-full border border-ink/10 px-3 py-1.5 font-mono text-xs">
                npm i @huggingface/kernels@preview
              </code>
            </div>
            <CodeBlock code={QUICK_START} />
            <p className="mt-5 font-mono text-[13px] leading-6 text-ink/55">
              The loader fetches the versioned contract and optimized WGSL from the Hub,
              validates inputs, selects a variant, and allocates the output.
            </p>
          </div>
        </div>
      </section>

      <section id="demos" className="bg-ink py-24 text-white sm:py-32">
        <AttentionDemo />
      </section>

      <section className="bg-cream py-24 text-ink sm:py-32">
        <MatrixWaves />
      </section>

      <footer className="bg-ink py-10 text-white">
        <div className="page-wrap flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="font-semibold">@huggingface/kernels</div>
          <p className="font-mono text-xs uppercase tracking-widest text-white/45">
            The first layer in making browser inference as fast as physically possible ·
            by{" "}
            <a
              href="https://huggingface.co/docs/transformers.js/en/index"
              target="_blank"
              rel="noreferrer"
              className="text-white/70 underline decoration-white/25 underline-offset-4 transition hover:text-white"
            >
              Transformers.js
            </a>
          </p>
        </div>
      </footer>
    </main>
  );
}
