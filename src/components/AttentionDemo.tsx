import { ArrowRight, BrainCircuit, ChevronDown, Sparkles } from "lucide-react";
import { useState } from "react";
import { Button } from "../theme";
import { runAttention } from "../utils/attention";
import type { AttentionResult } from "../utils/attention";
import cn from "../utils/classnames";
import CodeBlock from "./CodeBlock";

const ATTENTION_CODE = `import { getKernel } from "@huggingface/kernels";

// Load the four reusable GPU operations used by this attention block.
// getKernel fetches their WGSL implementations and prepares them to run.
const loadKernels = (names) => Promise.all(
  names.map((name) => getKernel(
    \`webgpu-kernels/ai.onnx.\${name}\`,
    { version: 1 }
  ))
);

const [matmul, softmax, add, norm] = await loadKernels([
  "MatMul", "Softmax", "Add", "LayerNormalization"
]);

// Asking for GPU output keeps intermediate tensors in GPU memory.
// They can flow directly into the next kernel without a CPU readback.
const gpu = { output: "gpu" };

// X contains one 128-number embedding for every input token.
// embed() adds BERT's token, position, and token-type embeddings.
// Shape: [number of tokens, 128 hidden features].
const X = embed(tokens, bertWeights);
const { Wq, Wk, Wv, Wo } = selectHead(bertWeights, 1);

// Learned weight matrices turn each token embedding into three views:
// Q (query): what information is this token looking for?
// K (key):   what information does this token contain?
// V (value): what information should this token contribute?
// Wq, Wk, and Wv each map 128 hidden features to one 64-feature head.
// Q is pre-scaled by 1 / sqrt(64), which prevents extreme score values.
const Q = await matmul({ a: X, b: Wq }, gpu);
const K = await matmul({ a: X, b: Wk }, gpu);
const V = await matmul({ a: X, b: Wv }, gpu);

// Compare every query token with every key token using dot products.
// transpose(K) only swaps K's two axes; it does not change its values.
// Q is [tokens, 64] and transpose(K) is [64, tokens], so scores is
// [tokens, tokens]. Row i says how strongly token i matches every token.
const scores = await matmul({ a: Q, b: transpose(K) }, gpu);

// Softmax converts each score row into probabilities that sum to 1.
// These are the attention weights displayed in the heatmap: rows are
// querying tokens and columns are the tokens they can pay attention to.
const attention = await softmax(
  { x: scores },
  { attrs: { axis: -1 }, output: "gpu" }
);

// Use those probabilities to blend the value vectors. The context row
// for each token is therefore a weighted summary of all input tokens.
// [tokens, tokens] × [tokens, 64] → [tokens, 64].
const context = await matmul({ a: attention, b: V }, gpu);

// Project the 64-feature head back to BERT's 128 hidden features.
const projected = await matmul({ a: context, b: Wo }, gpu);

// A residual connection preserves the original token representation,
// while LayerNorm keeps feature magnitudes stable for the next layer.
const residual = await add({ a: projected, b: X }, gpu);
const output = await norm(
  { x: residual, scale, b: bias },
  { attrs: { axis: -1, epsilon: 1e-12 } }
);`;

const DEFAULT_SENTENCE = "The small robot carried the red book across the quiet library.";

export default function AttentionDemo() {
  const [sentence, setSentence] = useState(DEFAULT_SENTENCE);
  const [result, setResult] = useState<AttentionResult | null>(null);
  const [status, setStatus] = useState("Ready · downloads 17 MB once");
  const [loading, setLoading] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [activeCell, setActiveCell] = useState<[number, number] | null>(null);
  const [head, setHead] = useState(1);

  const buildAttention = async () => {
    if (!sentence.trim() || loading) return;
    setLoading(true);
    try {
      const nextResult = await runAttention(sentence, setStatus, head);
      setResult(nextResult);
      setActiveCell(null);
      setStatus(`Head ${head} · layer 0 · pretrained BERT`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Attention failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-wrap">
      <div className="mb-14 grid gap-8 lg:grid-cols-[1fr_auto] lg:items-end">
        <div>
          <p className="eyebrow text-mint opacity-100">Demo 01 · Transformer attention</p>
          <h2 className="section-title mt-5 max-w-225">
            Build attention in{" "}
            <span className="font-serif font-normal italic block text-coral">
              20 lines.
            </span>
          </h2>
        </div>
        <p className="max-w-sm text-sm leading-6 text-white/45">
          A real sentence, Google’s pretrained tiny BERT weights, and the same four
          operations that sit inside a Transformers.js pipeline.
        </p>
      </div>

      <div className="overflow-hidden rounded-[28px] border border-white/12 bg-[#101211] text-white shadow-[0_30px_80px_rgba(60,24,17,.13)]">
        <div className="grid lg:grid-cols-[360px_minmax(0,1fr)]">
          <div className="border-b border-white/10 p-6 lg:border-b-0 lg:border-r lg:p-8">
            <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.12em] text-white/45">
              <BrainCircuit size={15} className="text-mint" /> What happens inside
              pipeline()?
            </div>
            <h3 className="mt-6 text-2xl font-semibold tracking-tight">
              Watch each token decide where to look.
            </h3>
            <p className="mt-4 text-sm leading-6 text-white/48">
              Attention turns every token into a query, key, and value. Brighter squares
              mean the token on that row pays more attention to the token in that column.
            </p>

            <form
              className="mt-8"
              onSubmit={(event) => {
                event.preventDefault();
                void buildAttention();
              }}
            >
              <label className="font-mono text-[11px] uppercase tracking-[0.1em] text-white/50">
                Sentence
              </label>
              <textarea
                value={sentence}
                onChange={(event) => setSentence(event.target.value)}
                rows={4}
                maxLength={140}
                className="mt-2 w-full resize-none rounded-xl border border-white/12 bg-white/5 p-3 text-sm leading-6 text-white outline-none transition focus:border-mint/50"
              />
              <div className="mt-4">
                <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-white/50">
                  Attention head
                </span>
                <div className="mt-2 grid grid-cols-2 rounded-full bg-white/6 p-1">
                  {[0, 1].map((value) => (
                    <button
                      type="button"
                      key={value}
                      onClick={() => {
                        setHead(value);
                        setResult(null);
                        setStatus(`Head ${value} selected · build to compare`);
                      }}
                      className={cn(
                        "rounded-full px-3 py-2 font-mono text-[11px] uppercase tracking-[0.1em] transition",
                        head === value
                          ? "bg-mint font-semibold text-ink"
                          : "text-white/45 hover:text-white"
                      )}
                    >
                      Head {value}
                    </button>
                  ))}
                </div>
              </div>
              <Button type="submit" disabled={loading} className="mt-3 w-full">
                {loading ? "Building…" : "Build attention"} <ArrowRight size={14} />
              </Button>
            </form>

            <div className="mt-6 flex gap-2 border-t border-white/10 pt-5 font-mono text-[11px] leading-5 text-white/45">
              <Sparkles size={13} className="mt-0.5 shrink-0 text-coral" />
              <span>{status}</span>
            </div>
          </div>

          <div className="min-h-[520px] overflow-hidden p-4 sm:p-7">
            {result ? (
              <>
                <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-mono text-[11px] uppercase tracking-[0.11em] text-white/45">
                      Attention heatmap · query ↓ · key →
                    </p>
                    <p className="mt-1 text-sm text-white/60">
                      {activeCell
                        ? `${result.tokens[activeCell[0]]} → ${result.tokens[activeCell[1]]}: ${(
                            result.attention[
                              activeCell[0] * result.tokenCount + activeCell[1]
                            ] * 100
                          ).toFixed(1)}% attention`
                        : "Hover a square to inspect the connection"}
                    </p>
                  </div>
                </div>
                <div className="overflow-x-auto rounded-2xl border border-white/10 bg-black/20 p-3">
                  <div
                    className="grid min-w-max gap-1"
                    style={{
                      gridTemplateColumns: `86px repeat(${result.tokenCount}, 44px)`,
                    }}
                  >
                    <div />
                    {result.tokens.map((token, index) => (
                      <div
                        key={`column-${index}`}
                        className="flex h-16 items-end justify-center overflow-hidden pb-1 font-mono text-[10px] text-white/52"
                      >
                        <span className="-rotate-45 whitespace-nowrap">{token}</span>
                      </div>
                    ))}
                    {result.tokens.flatMap((token, row) => {
                      const rowValues = result.attention.subarray(
                        row * result.tokenCount,
                        (row + 1) * result.tokenCount
                      );
                      const rowMax = Math.max(...rowValues, 0.0001);
                      const rowMin = Math.min(...rowValues);
                      const rowRange = Math.max(rowMax - rowMin, 0.0001);
                      return [
                        <div
                          key={`row-${row}`}
                          className="flex items-center truncate pr-2 font-mono text-[10px] text-white/58"
                        >
                          {token}
                        </div>,
                        ...Array.from(rowValues, (value, column) => (
                          <button
                            type="button"
                            key={`${row}-${column}`}
                            onMouseEnter={() => setActiveCell([row, column])}
                            onFocus={() => setActiveCell([row, column])}
                            className={cn(
                              "size-11 rounded-md border transition hover:scale-105 hover:border-white/50 focus:outline-none focus:ring-1 focus:ring-mint",
                              activeCell?.[0] === row && activeCell[1] === column
                                ? "border-white/60"
                                : "border-white/5"
                            )}
                            style={{
                              backgroundColor: `rgb(255 116 87 / ${0.1 + ((value - rowMin) / rowRange) * 0.86})`,
                            }}
                            aria-label={`${token} attends to ${result.tokens[column]} with ${(value * 100).toFixed(1)} percent weight`}
                          />
                        )),
                      ];
                    })}
                  </div>
                </div>
                <div className="mt-4 grid gap-3 rounded-xl border border-white/10 bg-white/[0.035] p-4 text-sm leading-6 text-white/55 sm:grid-cols-[auto_1fr]">
                  <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-mint/70">
                    Reading head {result.head}
                  </span>
                  <p>
                    Each row sums to 100%. A bright diagonal means a token preserves its
                    own state; a bright <code className="text-white">[CLS]</code> column
                    means information is routed through BERT’s sentence token. These are
                    learned routing patterns, not importance or explanation scores.
                  </p>
                </div>
              </>
            ) : (
              <div className="grid h-full min-h-[470px] place-items-center rounded-2xl border border-dashed border-white/12 text-center">
                <div className="max-w-sm px-6">
                  <div className="mx-auto grid size-12 place-items-center rounded-full bg-mint text-ink">
                    <BrainCircuit size={20} />
                  </div>
                  <h3 className="mt-5 font-semibold">No diagram is precomputed.</h3>
                  <p className="mt-2 text-sm leading-6 text-white/50">
                    Run the sentence to fetch real model weights, tokenize it, and build
                    the heatmap locally with WebGPU kernels.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowCode((value) => !value)}
          className="flex w-full items-center justify-between border-t border-white/10 px-6 py-5 text-left font-mono text-xs uppercase tracking-[0.12em] text-white/60 hover:text-white"
        >
          <span>Show me the code · MatMul → Softmax → Add → LayerNorm</span>
          <ChevronDown size={15} className={cn("transition", showCode && "rotate-180")} />
        </button>
        {showCode && <CodeBlock code={ATTENTION_CODE} className="m-3 mt-0" />}
      </div>
    </div>
  );
}
