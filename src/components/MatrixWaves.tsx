import { ChevronDown, Gauge, Pause, Play, RefreshCw, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "../theme";
import cn from "../utils/classnames";
import {
  CpuWaveRunner,
  FIELD_CHANNELS,
  FIELD_SIZE,
  GpuWaveRunner,
  paintWaveField,
} from "../utils/matrixWaves";
import type { WaveRunner } from "../utils/matrixWaves";
import CodeBlock from "./CodeBlock";

const WAVE_CODE = `const matmul = await getKernel(
  "webgpu-kernels/ai.onnx.MatMul",
  { version: 1 }
);

// Build the matrix that changes on every animation frame.
// Each row represents one image y-coordinate. Its 128 values are
// the current amplitudes of 128 independently moving wave channels.
function animateWaveSignals(timeMs, speed = 1) {
  const signals = new Float32Array(1024 * 128);

  // requestAnimationFrame-style timestamps are milliseconds elapsed
  // since the page started. Turning that clock into a phase offset is
  // what makes the otherwise static sine and cosine waves move.
  const movement = timeMs * 0.00035 * speed;

  for (let y = 0; y < 1024; y++) {
    for (let channel = 0; channel < 128; channel++) {
      // Every channel gets a different frequency and animation rate.
      const frequency = 0.005 + ((channel * 29) % 53) * 0.00065;
      const phase = channel * 1.618
        + movement * (0.35 + (channel % 9) * 0.08);

      signals[y * 128 + channel] =
        Math.sin(y * frequency + phase)
        + Math.cos(y * frequency * 0.41 - phase * 0.72) * 0.4;
    }
  }

  return signals;
}

// The current animation timestamp; a later frame receives a larger
// value and therefore generates different wave phases.
const timeMs = performance.now();

// Matrix A: 1024 image rows × 128 animated wave amplitudes.
const signals = animateWaveSignals(timeMs);

// Matrix B: 128 wave channels × 1024 image columns.
// This matrix is calculated once. Each channel stores a horizontal
// wave pattern and remains in GPU memory between animation frames.
const basis = precomputedWaveBasis;

// Matrix multiplication combines every vertical signal with every
// horizontal basis value. Each output value is the dot product of
// 128 channels, producing the complete 1024 × 1024 field in one call.
const { y: pixels } = await matmul({
  a: { data: signals, shape: [1024, 128] },
  b: basis, // GPU-resident
});

// Finally, map the floating-point field values to RGBA colors and
// write them into the canvas. This coloring step is not part of MatMul.
paintWaveField(canvas, pixels);`;

type Mode = "gpu" | "js";

export default function MatrixWaves() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runnerRef = useRef<WaveRunner | null>(null);
  const runningRef = useRef(true);
  const speedRef = useRef(1);
  const contrastRef = useRef(0.72);
  const generationRef = useRef(0);
  const [mode, setMode] = useState<Mode>("gpu");
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [contrast, setContrast] = useState(0.72);
  const [framesPerSecond, setFramesPerSecond] = useState(0);
  const [workloadTime, setWorkloadTime] = useState(0);
  const [status, setStatus] = useState("Loading MatMul from the Hub…");
  const [showCode, setShowCode] = useState(false);
  const [resetKey, setResetKey] = useState(0);

  useEffect(() => {
    runningRef.current = running;
    speedRef.current = speed;
    contrastRef.current = contrast;
  }, [running, speed, contrast]);

  useEffect(() => {
    const generation = ++generationRef.current;
    let disposed = false;
    let animationFrame = 0;
    runnerRef.current?.destroy();
    runnerRef.current = null;

    const run = async () => {
      setStatus(
        mode === "gpu" ? "Loading MatMul from the Hub…" : "134M JS operations per frame"
      );
      try {
        const runner =
          mode === "gpu" ? await GpuWaveRunner.create() : new CpuWaveRunner();
        if (disposed || generation !== generationRef.current) {
          runner.destroy();
          return;
        }
        runnerRef.current = runner;
        setStatus(
          mode === "gpu"
            ? "One MatMul · 1,048,576 output pixels"
            : "Three nested loops · main thread"
        );
        let renderedFrames = 0;
        let accumulatedWorkloadTime = 0;
        let metricStart = performance.now();

        const tick = async () => {
          if (disposed || generation !== generationRef.current) return;
          if (!runningRef.current) {
            animationFrame = requestAnimationFrame(() => void tick());
            return;
          }
          const started = performance.now();
          const result = await runner.frame(started, speedRef.current);
          accumulatedWorkloadTime += result.workloadTime;
          if (canvasRef.current) {
            paintWaveField(canvasRef.current, result.values, contrastRef.current);
          }
          renderedFrames += 1;
          const now = performance.now();
          if (now - metricStart > 800) {
            setFramesPerSecond((renderedFrames * 1000) / (now - metricStart));
            setWorkloadTime(accumulatedWorkloadTime / renderedFrames);
            renderedFrames = 0;
            accumulatedWorkloadTime = 0;
            metricStart = now;
          }
          animationFrame = requestAnimationFrame(() => void tick());
        };
        await tick();
      } catch (error) {
        setStatus(
          error instanceof Error ? error.message : "Could not start the wave field."
        );
        setRunning(false);
      }
    };
    void run();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      runnerRef.current?.destroy();
      runnerRef.current = null;
    };
  }, [mode, resetKey]);

  return (
    <div className="page-wrap">
      <div className="mb-14 grid gap-8 lg:grid-cols-[1fr_auto] lg:items-end">
        <div>
          <p className="eyebrow text-coral opacity-100">Demo 02 · Matrix waves</p>
          <h2 className="section-title mt-5 max-w-[900px]">
            One matrix multiply. One million moving pixels.
          </h2>
        </div>
        <p className="max-w-sm text-sm leading-6 text-ink/55">
          128 waves cross every row and column. MatMul combines them into a 1024 × 1024
          image in one kernel call; plain JavaScript runs the same 134 million
          calculations.
        </p>
      </div>

      <div className="overflow-hidden rounded-[28px] border border-ink/15 bg-[#101211] text-white shadow-[0_30px_80px_rgba(23,35,28,.1)]">
        <div className="grid lg:grid-cols-[minmax(0,1fr)_340px]">
          <div className="checker relative min-h-[420px] overflow-hidden sm:min-h-[650px]">
            <canvas
              ref={canvasRef}
              width={FIELD_SIZE}
              height={FIELD_SIZE}
              className="absolute inset-0 size-full object-cover"
              aria-label="Animated million-pixel matrix wave field"
            />
            <div className="absolute left-4 top-4 flex items-start gap-2 rounded-2xl border border-white/12 bg-black/55 px-3 py-2.5 font-mono text-[11px] uppercase tracking-[0.1em] backdrop-blur-md sm:left-6 sm:top-6">
              <span
                className={cn(
                  "mt-1 size-1.5 rounded-full",
                  running ? "bg-mint" : "bg-white/30"
                )}
              />
              <span>
                <span className="block text-white">
                  {framesPerSecond
                    ? `${framesPerSecond.toFixed(1)} display frames / sec`
                    : "Warming up"}
                </span>
                {workloadTime > 0 && (
                  <span className="mt-1 block tracking-[0.08em] text-mint/75">
                    {workloadTime.toFixed(2)} ms{" "}
                    {mode === "gpu" ? "kernel round trip" : "JS work"} · ~
                    {Math.round(1000 / workloadTime)} / sec uncapped
                  </span>
                )}
              </span>
            </div>
          </div>

          <aside className="flex flex-col border-t border-white/10 p-6 lg:border-l lg:border-t-0 lg:p-7">
            <div className="grid grid-cols-2 rounded-full bg-white/6 p-1">
              {(["gpu", "js"] as const).map((value) => (
                <button
                  type="button"
                  key={value}
                  onClick={() => {
                    setRunning(true);
                    setFramesPerSecond(0);
                    setWorkloadTime(0);
                    setMode(value);
                  }}
                  className={cn(
                    "rounded-full px-3 py-2.5 font-mono text-[11px] uppercase tracking-[0.1em] transition",
                    mode === value
                      ? "bg-coral font-semibold text-ink"
                      : "text-white/45 hover:text-white"
                  )}
                >
                  {value === "gpu" ? "GPU kernel" : "Plain JS"}
                </button>
              ))}
            </div>

            <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.035] p-4 text-sm leading-6 text-white/60">
              Each output pixel is the dot product of 128 moving wave values. The GPU
              calculates all pixels in parallel; JavaScript visits them one by one.
            </div>

            <div className="mt-8 space-y-7">
              <label className="block">
                <span className="mb-3 flex justify-between font-mono text-[11px] uppercase tracking-[0.09em] text-white/55">
                  <span>Motion</span>
                  <span className="text-white">{speed.toFixed(1)}×</span>
                </span>
                <input
                  className="w-full"
                  type="range"
                  min="0.2"
                  max="2.5"
                  step="0.1"
                  value={speed}
                  onChange={(event) => setSpeed(Number(event.target.value))}
                />
              </label>
              <label className="block">
                <span className="mb-3 flex justify-between font-mono text-[11px] uppercase tracking-[0.09em] text-white/55">
                  <span>Contrast</span>
                  <span className="text-white">{contrast.toFixed(2)}</span>
                </span>
                <input
                  className="w-full"
                  type="range"
                  min="0.2"
                  max="1.5"
                  step="0.01"
                  value={contrast}
                  onChange={(event) => setContrast(Number(event.target.value))}
                />
              </label>
            </div>

            <div className="mt-8 grid grid-cols-2 gap-2">
              <Button variant="light" onClick={() => setRunning((value) => !value)}>
                {running ? <Pause size={14} /> : <Play size={14} />}
                {running ? "Pause" : "Run"}
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  setRunning(true);
                  setFramesPerSecond(0);
                  setWorkloadTime(0);
                  setResetKey((value) => value + 1);
                }}
              >
                <RefreshCw size={14} /> Reset
              </Button>
            </div>

            <div className="mt-auto pt-10">
              <div className="flex gap-3 border-t border-white/10 pt-5">
                {mode === "gpu" ? (
                  <Zap className="mt-0.5 text-coral" size={16} />
                ) : (
                  <Gauge className="mt-0.5 text-white/45" size={16} />
                )}
                <p className="font-mono text-[11px] leading-5 text-white/48">{status}</p>
              </div>
              <p className="mt-3 font-mono text-[11px] text-white/35">
                [{FIELD_SIZE}, {FIELD_CHANNELS}] × [{FIELD_CHANNELS}, {FIELD_SIZE}]
              </p>
            </div>
          </aside>
        </div>

        <button
          type="button"
          onClick={() => setShowCode((value) => !value)}
          className="flex w-full items-center justify-between border-t border-white/10 px-6 py-5 text-left font-mono text-xs uppercase tracking-[0.12em] text-white/60 hover:text-white"
        >
          <span>Show me the code · one kernel call per frame</span>
          <ChevronDown size={15} className={cn("transition", showCode && "rotate-180")} />
        </button>
        {showCode && <CodeBlock code={WAVE_CODE} className="m-3 mt-0" />}
      </div>
    </div>
  );
}
