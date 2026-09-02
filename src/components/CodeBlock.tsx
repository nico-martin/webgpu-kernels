import { Check, Copy } from "lucide-react";
import Prism from "prismjs";
import "prismjs/components/prism-javascript";
import { useState } from "react";

interface CodeBlockProps {
  code: string;
  language?: "javascript" | "wgsl" | "jinja";
  className?: string;
}

Prism.languages.wgsl = {
  comment: /\/\*[\s\S]*?\*\/|\/\/.*$/m,
  annotation: { pattern: /@[a-z_]+/i, alias: "function" },
  keyword:
    /\b(?:fn|var|let|const|struct|return|if|else|for|while|loop|break|continue|enable|override)\b/,
  type: /\b(?:array|atomic|bool|f16|f32|i32|u32|vec[234]|mat[234]x[234])\b/,
  number: /\b(?:0x[\da-f]+|\d*\.?\d+(?:e[+-]?\d+)?)[fiu]?\b/i,
  operator: /->|&&|\|\||[+*/%!=<>-]=?|[&|^~]/,
  punctuation: /[{}[\];(),.:]/,
};

export default function CodeBlock({
  code,
  language = "javascript",
  className = "",
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const grammar =
    language === "javascript" ? Prism.languages.javascript : Prism.languages.wgsl;
  const highlightedCode = Prism.highlight(code, grammar, language);
  const label =
    language === "wgsl" ? "WGSL" : language === "jinja" ? "WGSL · Jinja" : "JavaScript";

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-code text-sm ${className}`}>
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-3 font-mono text-[11px] uppercase tracking-[0.16em] text-white/50">
        <span>{label}</span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-2 transition hover:text-white"
          aria-label="Copy code"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-5 font-mono text-[12px] leading-6 text-white/78 sm:p-7 sm:text-[13px]">
        <code
          className={`code-highlight language-${language}`}
          dangerouslySetInnerHTML={{ __html: highlightedCode }}
        />
      </pre>
    </div>
  );
}
