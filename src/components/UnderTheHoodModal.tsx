import { ArrowRight, Braces, Cpu, GitBranch, Layers3, X, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "../theme";
import CodeBlock from "./CodeBlock";

const SIMPLE_WGSL = `struct Params { length: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.length) {
    out[id.x] = a[id.x] + b[id.x];
  }
}`;

const JINJA_WGSL = `@group(0) @binding(0)
var<storage, read> a: array<{{ ELEMENT_TYPE }}>;
@group(0) @binding(1)
var<storage, read> b: array<{{ ELEMENT_TYPE }}>;
@group(0) @binding(2)
var<storage, read_write> out: array<{{ ELEMENT_TYPE }}>;

@compute @workgroup_size({{ WORKGROUP_SIZE }})
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.length / {{ VECTOR_WIDTH }}u) {
    out[id.x] = a[id.x] + b[id.x];
  }
}`;

const SCALAR_VARIANT = `// Broad compatibility · scalar lanes
var<storage, read> a: array<f32>;
var<storage, read> b: array<f32>;
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.length) {
    out[id.x] = a[id.x] + b[id.x];
  }
}`;

const VECTOR_VARIANT = `// Wide device · four values per invocation
var<storage, read> a: array<vec4<f32>>;
var<storage, read> b: array<vec4<f32>>;
var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.length / 4u) {
    out[id.x] = a[id.x] + b[id.x];
  }
}`;

const LOADER_CODE = `import { getKernel } from "@huggingface/kernels";

const add = await getKernel(
  "webgpu-kernels/ai.onnx.Add",
  { version: 1 }
);

const { c } = await add({
  a: { data: new Float32Array([1, 2, 3]), shape: [3] },
  b: { data: new Float32Array([10]), shape: [1] },
});

// Contract validation, device inspection, Jinja rendering,
// pipeline creation, dispatch, and readback happened above.`;

interface UnderTheHoodModalProps {
  className?: string;
  label?: string;
  variant?: "primary" | "ghost" | "light";
}

export default function UnderTheHoodModal({
  className = "",
  label = "Go under the hood",
  variant = "primary",
}: UnderTheHoodModalProps) {
  const [open, setOpen] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <>
      <Button className={className} variant={variant} onClick={() => setOpen(true)}>
        {label} <ArrowRight size={15} />
      </Button>

      {open && (
        <div
          className="fixed inset-0 z-[100] grid place-items-center bg-black/75 p-2 backdrop-blur-sm sm:p-5"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setOpen(false);
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="under-the-hood-title"
            className="relative flex max-h-[94dvh] w-full max-w-[1180px] flex-col overflow-hidden rounded-[24px] bg-cream text-ink shadow-2xl sm:rounded-[32px]"
          >
            <header className="z-10 flex shrink-0 items-center justify-between border-b border-ink/10 bg-cream px-5 py-4 sm:px-8">
              <div className="flex items-center gap-3">
                <span className="grid size-9 place-items-center rounded-full bg-ink text-mint">
                  <Cpu size={17} />
                </span>
                <div>
                  <p className="eyebrow">Under the hood</p>
                  <p className="text-sm font-semibold">
                    From an operation to GPU execution
                  </p>
                </div>
              </div>
              <button
                ref={closeRef}
                type="button"
                onClick={() => setOpen(false)}
                className="grid size-10 place-items-center rounded-full border border-ink/15 transition hover:bg-ink hover:text-white"
                aria-label="Close modal"
              >
                <X size={18} />
              </button>
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 py-10 sm:px-8 lg:px-12 lg:py-14">
              <div className="max-w-3xl">
                <p className="eyebrow text-coral opacity-100">
                  Five layers · one async call
                </p>
                <h2
                  id="under-the-hood-title"
                  className="mt-4 text-[clamp(2.6rem,6vw,5.8rem)] font-semibold leading-[0.9] tracking-[-0.065em]"
                >
                  How JavaScript becomes GPU work.
                </h2>
                <p className="body-copy mt-7">
                  The one-line API hides a deliberate pipeline: define the math, inspect
                  the device, choose an implementation, render a shader, and dispatch it.
                </p>
              </div>

              <div className="mt-16 space-y-20">
                <article className="grid gap-8 lg:grid-cols-[240px_1fr]">
                  <ModalStep
                    number="01"
                    icon={Layers3}
                    title="Ops are intent. Kernels are execution."
                  />
                  <div>
                    <p className="text-lg leading-8 text-ink/65">
                      An <strong className="text-ink">operation</strong> describes the
                      math: inputs, outputs, shapes, types, and semantics. “Add these
                      broadcastable tensors” means the same thing everywhere. A{" "}
                      <strong className="text-ink">kernel</strong> is a concrete program
                      that performs that operation on a particular execution platform.
                    </p>
                    <div className="mt-7 grid gap-px overflow-hidden rounded-2xl border border-ink/10 bg-ink/10 sm:grid-cols-2">
                      <div className="bg-cream p-6">
                        <p className="eyebrow">Operation · Add</p>
                        <p className="mt-5 font-serif text-3xl italic">C = A + B</p>
                        <p className="mt-3 text-sm leading-6 text-ink/55">
                          Portable contract. No workgroup size, language, or device
                          assumptions.
                        </p>
                      </div>
                      <div className="bg-cream p-6">
                        <p className="eyebrow">Kernel · WebGPU</p>
                        <p className="mt-5 font-mono text-lg">out[i] = a[i] + b[i]</p>
                        <p className="mt-3 text-sm leading-6 text-ink/55">
                          Executable implementation in WGSL. CUDA or Metal would use
                          another kernel.
                        </p>
                      </div>
                    </div>
                  </div>
                </article>

                <article className="grid gap-8 lg:grid-cols-[240px_1fr]">
                  <ModalStep
                    number="02"
                    icon={Braces}
                    title="WGSL sits between JavaScript and the GPU."
                  />
                  <div>
                    <p className="mb-7 text-lg leading-8 text-ink/65">
                      JavaScript prepares buffers and starts the work. WGSL is the shader
                      language the browser validates and compiles for the current GPU.
                      This minimal compute kernel assigns one array element to each
                      invocation.
                    </p>
                    <CodeBlock code={SIMPLE_WGSL} language="wgsl" />
                  </div>
                </article>

                <article className="grid gap-8 lg:grid-cols-[240px_1fr]">
                  <ModalStep
                    number="03"
                    icon={GitBranch}
                    title="Fast is device-specific."
                  />
                  <div>
                    <p className="text-lg leading-8 text-ink/65">
                      GPUs differ in workgroup limits, memory bandwidth, supported data
                      types, and subgroup behavior. A shader tuned for a desktop GPU may
                      lose badly on integrated hardware. There is no universally fastest
                      implementation.
                    </p>
                    <div className="mt-7 rounded-2xl bg-ink p-6 text-white sm:p-8">
                      <div className="grid gap-8 sm:grid-cols-3">
                        {[
                          ["64–1024", "workgroup limits"],
                          ["f16?", "optional features"],
                          ["4–128", "subgroup widths"],
                        ].map(([value, label]) => (
                          <div key={label}>
                            <p className="font-mono text-2xl text-mint">{value}</p>
                            <p className="mt-2 text-sm text-white/50">{label}</p>
                          </div>
                        ))}
                      </div>
                      <p className="mt-7 border-t border-white/10 pt-6 text-sm leading-6 text-white/55">
                        <strong className="text-white">Subgroups</strong> let nearby GPU
                        lanes exchange data without round-tripping through shared memory.
                        Variants let a kernel use those capabilities where they help and
                        retain a safe path everywhere else.
                      </p>
                    </div>
                  </div>
                </article>

                <article className="grid gap-8 lg:grid-cols-[240px_1fr]">
                  <ModalStep
                    number="04"
                    icon={Zap}
                    title="Jinja builds the right WGSL at runtime."
                  />
                  <div>
                    <p className="text-lg leading-8 text-ink/65">
                      Published shaders are templates, not frozen strings. Device facts
                      and the selected variant become Jinja variables. The loader renders
                      only the WGSL needed for this call and this GPU.
                    </p>
                    <CodeBlock code={JINJA_WGSL} language="jinja" className="mt-7" />
                    <div className="mt-4 grid gap-4 xl:grid-cols-2">
                      <CodeBlock code={SCALAR_VARIANT} language="wgsl" />
                      <CodeBlock code={VECTOR_VARIANT} language="wgsl" />
                    </div>
                    <p className="mt-4 text-sm leading-6 text-ink/55">
                      Both rendered shaders implement Add. The scalar version prioritizes
                      broad compatibility; the vectorized version processes four values
                      per invocation and uses a larger workgroup when the device and shape
                      make that profitable.
                    </p>
                  </div>
                </article>

                <article className="grid gap-8 lg:grid-cols-[240px_1fr]">
                  <ModalStep
                    number="05"
                    icon={Cpu}
                    title="@huggingface/kernels glues it together."
                  />
                  <div>
                    <div className="mb-7 grid gap-2 sm:grid-cols-5">
                      {["Load", "Validate", "Select", "Render", "Run"].map(
                        (item, index) => (
                          <div
                            key={item}
                            className="flex items-center justify-between rounded-xl border border-ink/10 px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider"
                          >
                            {item}
                            {index < 4 && (
                              <ArrowRight
                                size={13}
                                className="hidden opacity-30 sm:block"
                              />
                            )}
                          </div>
                        )
                      )}
                    </div>
                    <p className="mb-7 text-lg leading-8 text-ink/65">
                      The library fetches the versioned manifest and Jinja files, checks
                      the operation contract, inspects WebGPU capabilities, selects a
                      valid variant, renders and compiles WGSL, manages buffers, and
                      returns a normal async JavaScript executable.
                    </p>
                    <CodeBlock code={LOADER_CODE} />
                  </div>
                </article>
              </div>
            </div>
          </section>
        </div>
      )}
    </>
  );
}

interface ModalStepProps {
  number: string;
  icon: typeof Cpu;
  title: string;
}

function ModalStep({ number, icon: Icon, title }: ModalStepProps) {
  return (
    <div>
      <div className="flex items-center gap-3">
        <span className="font-mono text-xs text-coral">{number}</span>
        <Icon size={18} strokeWidth={1.7} />
      </div>
      <h3 className="mt-4 text-2xl font-semibold leading-tight tracking-[-0.025em]">
        {title}
      </h3>
    </div>
  );
}
