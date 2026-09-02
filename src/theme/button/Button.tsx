import type { ButtonHTMLAttributes } from "react";
import cn from "../../utils/classnames";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "light";
  className?: string;
}

export default function Button({
  variant = "primary",
  className = "",
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex h-11 items-center justify-center gap-2 rounded-full px-5 font-mono text-xs font-semibold uppercase tracking-[0.12em] transition disabled:cursor-not-allowed disabled:opacity-45",
        {
          "bg-coral text-ink hover:bg-[#ff8b70]": variant === "primary",
          "border border-white/20 text-white hover:border-white/50 hover:bg-white/5":
            variant === "ghost",
          "bg-cream text-ink hover:bg-white": variant === "light",
        },
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
