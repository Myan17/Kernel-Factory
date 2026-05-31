import type { ReactNode } from "react";
import Reveal from "./Reveal";

interface Props {
  eyebrow: string;
  title: ReactNode;
  subtitle?: ReactNode;
  center?: boolean;
}

export default function SectionHeading({ eyebrow, title, subtitle, center }: Props) {
  return (
    <Reveal className={center ? "mx-auto max-w-2xl text-center" : "max-w-2xl"}>
      <div
        className={`mb-3 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 font-mono text-xs font-medium uppercase tracking-wider text-indigo ${
          center ? "" : ""
        }`}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-indigo" />
        {eyebrow}
      </div>
      <h2 className="font-display text-3xl font-extrabold leading-tight tracking-tight sm:text-4xl md:text-[2.7rem]">
        {title}
      </h2>
      {subtitle && <p className="mt-4 text-base leading-relaxed text-muted sm:text-lg">{subtitle}</p>}
    </Reveal>
  );
}
