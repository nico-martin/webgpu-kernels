import { getKernel } from "@huggingface/kernels";
import { MousePointer2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const FIELD_WIDTH = 128;
const FIELD_HEIGHT = 64;
const CHANNEL_COUNT = 12;
const TAU = Math.PI * 2;

interface Impact {
  direction: [number, number, number];
  u: number;
  v: number;
  started: number;
}

interface HeatRunner {
  frame(rows: Float32Array, columns: Float32Array): Promise<Float32Array>;
  destroy(): void;
}

class CpuHeatRunner implements HeatRunner {
  private readonly output = new Float32Array(FIELD_WIDTH * FIELD_HEIGHT);

  async frame(rows: Float32Array, columns: Float32Array): Promise<Float32Array> {
    this.output.fill(0);
    for (let y = 0; y < FIELD_HEIGHT; y += 1) {
      for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
        const rowValue = rows[y * CHANNEL_COUNT + channel];
        for (let x = 0; x < FIELD_WIDTH; x += 1) {
          this.output[y * FIELD_WIDTH + x] +=
            rowValue * columns[channel * FIELD_WIDTH + x];
        }
      }
    }
    return this.output;
  }

  destroy() {}
}

class GpuHeatRunner implements HeatRunner {
  private constructor(private readonly matmul: Awaited<ReturnType<typeof getKernel>>) {}

  static async create(): Promise<GpuHeatRunner> {
    if (!("gpu" in navigator)) throw new Error("WebGPU unavailable");
    return new GpuHeatRunner(
      await getKernel("webgpu-kernels/ai.onnx.MatMul", { version: 1 })
    );
  }

  async frame(rows: Float32Array, columns: Float32Array): Promise<Float32Array> {
    const result = await this.matmul({
      a: { data: rows, shape: [FIELD_HEIGHT, CHANNEL_COUNT] },
      b: { data: columns, shape: [CHANNEL_COUNT, FIELD_WIDTH] },
    });
    const tensor = Object.values(result)[0];
    if (!tensor || !(tensor.data instanceof Float32Array)) {
      throw new Error("MatMul returned no heat field.");
    }
    return tensor.data;
  }

  destroy() {}
}

function wrappedDistance(a: number, b: number): number {
  const distance = Math.abs(a - b);
  return Math.min(distance, 1 - distance);
}

function fillHeatMatrices(
  rows: Float32Array,
  columns: Float32Array,
  time: number,
  now: number,
  impacts: Impact[]
) {
  rows.fill(0);
  columns.fill(0);
  const baseChannels = CHANNEL_COUNT - 4;

  for (let channel = 0; channel < baseChannels; channel += 1) {
    const frequency = channel + 1;
    for (let y = 0; y < FIELD_HEIGHT; y += 1) {
      const v = y / FIELD_HEIGHT;
      rows[y * CHANNEL_COUNT + channel] =
        Math.sin(v * TAU * (1 + (channel % 3)) + time * (0.18 + channel * 0.035)) *
        (0.11 / Math.sqrt(frequency));
    }
    for (let x = 0; x < FIELD_WIDTH; x += 1) {
      const u = x / FIELD_WIDTH;
      columns[channel * FIELD_WIDTH + x] =
        Math.cos(u * TAU * (2 + channel) - time * (0.12 + channel * 0.028)) +
        Math.sin((u * (channel + 3) + time * 0.025) * TAU) * 0.38;
    }
  }

  impacts.slice(-4).forEach((impact, impactIndex) => {
    const age = now - impact.started;
    if (age < 0 || age > 4.5) return;
    const channel = baseChannels + impactIndex;
    const envelope = Math.exp(-age * 0.82) * 1.8;
    const radius = 0.045 + age * 0.018;
    for (let y = 0; y < FIELD_HEIGHT; y += 1) {
      const distance = (y / FIELD_HEIGHT - impact.v) / radius;
      rows[y * CHANNEL_COUNT + channel] = Math.exp(-distance * distance * 1.8) * envelope;
    }
    for (let x = 0; x < FIELD_WIDTH; x += 1) {
      const distance = wrappedDistance(x / FIELD_WIDTH, impact.u) / radius;
      columns[channel * FIELD_WIDTH + x] = Math.exp(-distance * distance * 1.8);
    }
  });
}

const VERTEX_SHADER = `#version 300 es
void main() {
  vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uHeat;
uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uImpactDirection;
uniform float uImpactAge;
out vec4 outColor;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

float hash31(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float noise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash31(i), hash31(i + vec3(1,0,0)), f.x),
        mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y),
    mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
        mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y), f.z
  );
}

float fbm(vec3 p) {
  float value = 0.0;
  float amplitude = 0.5;
  for (int octave = 0; octave < 4; octave++) {
    value += noise3(p) * amplitude;
    p = p * 2.03 + vec3(0.17, 0.31, 0.13);
    amplitude *= 0.5;
  }
  return value;
}

float sphereIntersection(vec3 origin, vec3 direction) {
  float b = dot(origin, direction);
  float c = dot(origin, origin) - 1.0;
  float h = b * b - c;
  return h < 0.0 ? -1.0 : -b - sqrt(h);
}

void main() {
  vec2 q = (gl_FragCoord.xy * 2.0 - uResolution) / uResolution.y;
  vec3 origin = vec3(0.0, 0.0, 3.15);
  vec3 direction = normalize(vec3(q, -2.25));
  float hit = sphereIntersection(origin, direction);

  float screenRadius = length(q);
  float halo = exp(-max(screenRadius - 0.72, 0.0) * 15.0)
    * (1.0 - smoothstep(0.72, 1.05, screenRadius));
  vec3 color = vec3(0.32, 0.015, 0.008) * halo * 0.24;
  float alpha = halo * 0.28;

  if (hit > 0.0) {
    vec3 position = origin + direction * hit;
    vec3 normal = normalize(position);
    vec2 uv = vec2(
      fract(atan(normal.z, normal.x) / TAU + 0.5),
      asin(clamp(normal.y, -1.0, 1.0)) / PI + 0.5
    );
    float field = texture(uHeat, uv).r;
    float detail = fbm(normal * 7.0 + vec3(uTime * 0.035, -uTime * 0.02, 0.0));
    float veins = abs(fbm(normal * 15.0 - vec3(0.0, uTime * 0.04, 0.0)) - 0.5);
    float heat = clamp(0.34 + field + (detail - 0.5) * 0.28 + (0.11 - veins) * 0.28, 0.0, 1.3);

    float impactDistance = acos(clamp(dot(normal, uImpactDirection), -1.0, 1.0));
    float shockRadius = uImpactAge * 0.95;
    float shock = exp(-pow((impactDistance - shockRadius) * 18.0, 2.0))
      * exp(-uImpactAge * 0.72);
    float core = exp(-impactDistance * impactDistance * 95.0)
      * exp(-uImpactAge * 1.55);
    heat = clamp(heat + shock * 1.15 + core * 1.5, 0.0, 1.5);

    vec3 coldRock = vec3(0.012, 0.004, 0.003);
    vec3 warmRock = vec3(0.11, 0.012, 0.006);
    vec3 deepLava = vec3(0.88, 0.018, 0.004);
    vec3 orangeLava = vec3(1.0, 0.21, 0.025);
    vec3 whiteLava = vec3(1.0, 0.78, 0.25);
    float lavaMask = smoothstep(0.48, 0.62, heat);
    lavaMask = max(lavaMask, shock + core);
    vec3 lava = mix(deepLava, orangeLava, smoothstep(0.35, 0.78, heat));
    lava = mix(lava, whiteLava, smoothstep(0.86, 1.35, heat));

    vec3 lightDirection = normalize(vec3(-0.5, 0.72, 0.62));
    vec3 viewDirection = -direction;
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);
    float specular = pow(max(dot(reflect(-lightDirection, normal), viewDirection), 0.0), 52.0);
    vec3 rock = mix(coldRock, warmRock, detail) * (0.18 + diffuse * 1.25);
    color = mix(rock, lava * (1.0 + heat * 1.35), lavaMask);
    color += orangeLava * fresnel * (0.09 + lavaMask * 0.4);
    color += whiteLava * specular * (0.08 + lavaMask * 0.28);
    color = 1.0 - exp(-color * 1.08);
    color = pow(color, vec3(0.4545));
    alpha = 1.0;
  }

  outColor = vec4(color, alpha);
}`;

function createShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not create sphere shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "Sphere shader compilation failed.");
  }
  return shader;
}

class SphereRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly texture: WebGLTexture;
  private readonly vao: WebGLVertexArrayObject;
  private readonly uniforms: Record<string, WebGLUniformLocation>;
  private impactDirection: [number, number, number] = [0, 0, 1];
  private impactAt = -100;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      powerPreference: "high-performance",
      premultipliedAlpha: true,
    });
    if (!gl) throw new Error("WebGL 2 is unavailable.");
    this.gl = gl;
    const program = gl.createProgram();
    const vao = gl.createVertexArray();
    const texture = gl.createTexture();
    if (!program || !vao || !texture)
      throw new Error("Could not allocate sphere renderer.");
    this.program = program;
    this.vao = vao;
    this.texture = texture;
    const vertex = createShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = createShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) ?? "Could not link sphere shaders.");
    }

    const names = ["uHeat", "uResolution", "uTime", "uImpactDirection", "uImpactAge"];
    this.uniforms = Object.fromEntries(
      names.map((name) => {
        const location = gl.getUniformLocation(program, name);
        if (!location) throw new Error(`Missing sphere uniform ${name}.`);
        return [name, location];
      })
    );
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const filter = gl.getExtension("OES_texture_float_linear") ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      FIELD_WIDTH,
      FIELD_HEIGHT,
      0,
      gl.RED,
      gl.FLOAT,
      null
    );
    gl.uniform1i(this.uniforms.uHeat, 0);
    gl.disable(gl.DEPTH_TEST);
  }

  upload(field: Float32Array) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      FIELD_WIDTH,
      FIELD_HEIGHT,
      gl.RED,
      gl.FLOAT,
      field
    );
  }

  render(time: number, now: number) {
    const gl = this.gl;
    const ratio = Math.min(window.devicePixelRatio || 1, 1.7);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform2f(this.uniforms.uResolution, width, height);
    gl.uniform1f(this.uniforms.uTime, time);
    gl.uniform3fv(this.uniforms.uImpactDirection, this.impactDirection);
    gl.uniform1f(this.uniforms.uImpactAge, Math.max(0, now - this.impactAt));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  pick(clientX: number, clientY: number): Impact | null {
    const bounds = this.canvas.getBoundingClientRect();
    const qx = ((clientX - bounds.left) / bounds.width) * 2 - 1;
    const qy = 1 - ((clientY - bounds.top) / bounds.height) * 2;
    const origin: [number, number, number] = [0, 0, 3.15];
    const length = Math.hypot(qx, qy, 2.25);
    const direction: [number, number, number] = [
      qx / length,
      qy / length,
      -2.25 / length,
    ];
    const b = origin[2] * direction[2];
    const c = origin[2] ** 2 - 1;
    const discriminant = b * b - c;
    if (discriminant < 0) return null;
    const distance = -b - Math.sqrt(discriminant);
    if (distance <= 0) return null;
    const surface: [number, number, number] = [
      origin[0] + direction[0] * distance,
      origin[1] + direction[1] * distance,
      origin[2] + direction[2] * distance,
    ];
    const surfaceLength = Math.hypot(...surface);
    const normal: [number, number, number] = surface.map(
      (value) => value / surfaceLength
    ) as [number, number, number];
    const started = performance.now() / 1000;
    return {
      direction: normal,
      u: (((Math.atan2(normal[2], normal[0]) / TAU + 0.5) % 1) + 1) % 1,
      v: Math.asin(Math.max(-1, Math.min(1, normal[1]))) / Math.PI + 0.5,
      started,
    };
  }

  impact(impact: Impact) {
    this.impactDirection = impact.direction;
    this.impactAt = impact.started;
  }

  destroy() {
    this.gl.deleteTexture(this.texture);
    this.gl.deleteVertexArray(this.vao);
    this.gl.deleteProgram(this.program);
  }
}

export default function KernelBalloon() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<SphereRenderer | null>(null);
  const impactsRef = useRef<Impact[]>([]);
  const [backend, setBackend] = useState("Loading MatMul…");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let disposed = false;
    let animationFrame = 0;
    let runner: HeatRunner = new CpuHeatRunner();
    let renderer: SphereRenderer;
    try {
      renderer = new SphereRenderer(canvas);
      rendererRef.current = renderer;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Sphere renderer unavailable";
      queueMicrotask(() => {
        if (!disposed) setBackend(message);
      });
      return;
    }
    const rows = new Float32Array(FIELD_HEIGHT * CHANNEL_COUNT);
    const columns = new Float32Array(CHANNEL_COUNT * FIELD_WIDTH);
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    void GpuHeatRunner.create()
      .then((gpuRunner) => {
        if (disposed) {
          gpuRunner.destroy();
          return;
        }
        runner.destroy();
        runner = gpuRunner;
        setBackend("WebGPU MatMul");
      })
      .catch(() => setBackend("JavaScript fallback"));

    const tick = async (timestamp: number) => {
      if (disposed) return;
      const time = reducedMotion ? 0 : timestamp / 1000;
      impactsRef.current = impactsRef.current.filter(
        (impact) => timestamp / 1000 - impact.started < 4.5
      );
      fillHeatMatrices(rows, columns, time, timestamp / 1000, impactsRef.current);
      const field = await runner.frame(rows, columns);
      if (disposed) return;
      renderer.upload(field);
      renderer.render(time, timestamp / 1000);
      animationFrame = requestAnimationFrame((next) => void tick(next));
    };
    animationFrame = requestAnimationFrame((timestamp) => void tick(timestamp));

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      runner.destroy();
      renderer.destroy();
      rendererRef.current = null;
    };
  }, []);

  const erupt = (clientX?: number, clientY?: number) => {
    const renderer = rendererRef.current;
    const canvas = canvasRef.current;
    if (!renderer || !canvas) return;
    const bounds = canvas.getBoundingClientRect();
    const impact = renderer.pick(
      clientX ?? bounds.left + bounds.width * 0.66,
      clientY ?? bounds.top + bounds.height * 0.35
    );
    if (!impact) return;
    impactsRef.current.push(impact);
    renderer.impact(impact);
  };

  return (
    <div className="relative mx-auto w-full max-w-140 pb-12">
      <button
        type="button"
        className="group relative block aspect-square w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-coral"
        onClick={(event) => erupt(event.clientX || undefined, event.clientY || undefined)}
        aria-label="Trigger a kernel-driven eruption on the molten sphere"
      >
        <canvas ref={canvasRef} className="size-full cursor-crosshair" />
        <span className="absolute left-1/2 top-[47%] flex -translate-x-1/2 items-center gap-2 rounded-full border border-white/15 bg-black/35 px-3 py-2 font-mono text-[9px] uppercase tracking-[0.12em] text-white/60 opacity-70 backdrop-blur-md transition sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-visible:opacity-100">
          <MousePointer2 size={12} /> Trigger eruption
        </span>
      </button>

      <div className="absolute inset-x-[5%] bottom-0 grid grid-cols-3 overflow-hidden rounded-2xl border border-white/10 bg-black/70 backdrop-blur-xl">
        {[
          ["1", "MatMul / frame"],
          [String(CHANNEL_COUNT), "heat channels"],
          ["8,192", "surface cells"],
        ].map(([value, label]) => (
          <div className="border-r border-white/10 p-4 last:border-0 sm:p-5" key={label}>
            <div className="text-2xl font-semibold tracking-tight sm:text-3xl">
              {value}
            </div>
            <div className="mt-1 font-mono text-[9px] uppercase tracking-[0.11em] text-white/45 sm:text-[10px]">
              {label}
            </div>
          </div>
        ))}
      </div>
      <span className="absolute right-[7%] top-[8%] rounded-full border border-white/10 bg-black/30 px-3 py-1.5 font-mono text-[8px] uppercase tracking-wider text-white/40 backdrop-blur-sm">
        {backend}
      </span>
    </div>
  );
}
