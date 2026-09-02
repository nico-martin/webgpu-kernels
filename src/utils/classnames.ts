export default function cn(
  ...classes: Array<string | false | null | undefined | Record<string, boolean>>
): string {
  return classes
    .flatMap((value) => {
      if (typeof value === "string") return value;
      if (!value) return [];
      return Object.entries(value)
        .filter(([, enabled]) => enabled)
        .map(([className]) => className);
    })
    .join(" ");
}
