import { useState, type ReactNode } from "react";

/**
 * Collapsible control panel: a left side panel on desktop, a bottom sheet under 768 px.
 */
export default function Panel({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <aside
      aria-label="Map controls"
      className={[
        "pointer-events-auto absolute z-[1000] flex flex-col bg-slate-900/90 text-slate-100 shadow-2xl backdrop-blur",
        // Mobile: bottom sheet. Desktop: left panel.
        "inset-x-0 bottom-0 max-h-[60vh] rounded-t-2xl",
        "md:inset-x-auto md:bottom-auto md:left-4 md:top-4 md:max-h-[calc(100%-2rem)] md:w-80 md:rounded-2xl",
      ].join(" ")}
    >
      <header className="flex items-center justify-between gap-2 border-b border-slate-700/60 px-4 py-3">
        <div>
          <h1 className="text-base font-semibold tracking-tight">Deforestation Monitor</h1>
          <p className="text-xs text-slate-400">Sentinel-2 · U-Net · monthly change detection</p>
        </div>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls="panel-body"
          className="rounded-lg px-2 py-1 text-sm text-slate-300 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
        >
          {open ? "Hide" : "Show"}
        </button>
      </header>
      {/* Not the `hidden` attribute: Tailwind's `flex` display would override it. */}
      <div id="panel-body" className={`${open ? "flex" : "hidden"} flex-col gap-4 overflow-y-auto px-4 py-4`}>
        {children}
      </div>
    </aside>
  );
}
