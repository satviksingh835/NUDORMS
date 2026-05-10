import { useCallback, useState } from "react";
import { SplatViewer } from "../viewer/SplatViewer";

export function DemoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);

  const accept = useCallback((f: File) => {
    if (f.name.endsWith(".ply") || f.name.endsWith(".splat")) setFile(f);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) accept(f);
    },
    [accept],
  );

  const onFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) accept(f);
    },
    [accept],
  );

  if (file) {
    return (
      <div style={{ position: "fixed", inset: 0, background: "#0a0a0a" }}>
        <SplatViewer file={file} />

        {/* Back button */}
        <button
          onClick={() => setFile(null)}
          style={styles.backBtn}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Back
        </button>

        {/* File label */}
        <div style={styles.fileLabel}>
          <span style={styles.fileLabelDot} />
          {file.name}
        </div>

        <style>{fonts}</style>
      </div>
    );
  }

  return (
    <div style={styles.root}>
      <style>{fonts}</style>

      {/* Wordmark */}
      <div style={styles.wordmark}>NUDORMS</div>

      {/* Drop zone */}
      <div
        style={{
          ...styles.dropZone,
          ...(dragging ? styles.dropZoneActive : {}),
        }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div style={styles.dropIcon}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        <p style={styles.dropTitle}>Drop your file here</p>
        <p style={styles.dropSub}>.ply · .splat</p>

        <label style={styles.browseBtn}>
          Browse files
          <input
            type="file"
            accept=".ply,.splat"
            style={{ display: "none" }}
            onChange={onFileInput}
          />
        </label>
      </div>

      <p style={styles.footer}>
        Local preview · no upload · no backend required
      </p>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    position: "fixed",
    inset: 0,
    background: "#0a0a0a",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'DM Mono', monospace",
    gap: 0,
  },
  wordmark: {
    position: "fixed",
    top: 24,
    left: 24,
    color: "rgba(255,255,255,0.9)",
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.18em",
    fontFamily: "'DM Mono', monospace",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 999,
    padding: "7px 14px",
  },
  dropZone: {
    width: 380,
    padding: "56px 40px",
    border: "1px dashed rgba(255,255,255,0.12)",
    borderRadius: 20,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 12,
    background: "rgba(255,255,255,0.015)",
    transition: "border-color 0.2s, background 0.2s",
    cursor: "default",
  },
  dropZoneActive: {
    borderColor: "rgba(255,255,255,0.4)",
    background: "rgba(255,255,255,0.04)",
  },
  dropIcon: {
    color: "rgba(255,255,255,0.25)",
    marginBottom: 8,
  },
  dropTitle: {
    margin: 0,
    color: "rgba(255,255,255,0.75)",
    fontSize: 15,
    fontWeight: 500,
    letterSpacing: "0.03em",
  },
  dropSub: {
    margin: 0,
    color: "rgba(255,255,255,0.25)",
    fontSize: 11,
    letterSpacing: "0.08em",
  },
  browseBtn: {
    marginTop: 16,
    background: "rgba(255,255,255,0.07)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 999,
    color: "rgba(255,255,255,0.65)",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: "0.08em",
    padding: "8px 20px",
    cursor: "pointer",
    fontFamily: "'DM Mono', monospace",
    transition: "background 0.15s",
  },
  footer: {
    position: "fixed",
    bottom: 28,
    margin: 0,
    color: "rgba(255,255,255,0.15)",
    fontSize: 10,
    letterSpacing: "0.06em",
  },
  backBtn: {
    position: "fixed",
    top: 20,
    left: 20,
    zIndex: 30,
    display: "flex",
    alignItems: "center",
    gap: 6,
    background: "rgba(10,10,10,0.55)",
    backdropFilter: "blur(12px)",
    WebkitBackdropFilter: "blur(12px)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 999,
    color: "rgba(255,255,255,0.7)",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: "0.06em",
    fontFamily: "'DM Mono', monospace",
    padding: "7px 14px",
    cursor: "pointer",
  },
  fileLabel: {
    position: "fixed",
    top: 20,
    left: "50%",
    transform: "translateX(-50%)",
    zIndex: 30,
    display: "flex",
    alignItems: "center",
    gap: 8,
    background: "rgba(10,10,10,0.55)",
    backdropFilter: "blur(12px)",
    WebkitBackdropFilter: "blur(12px)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 999,
    color: "rgba(255,255,255,0.55)",
    fontSize: 11,
    letterSpacing: "0.06em",
    fontFamily: "'DM Mono', monospace",
    padding: "7px 16px",
    maxWidth: 320,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  fileLabelDot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "#4ade80",
    flexShrink: 0,
    boxShadow: "0 0 8px #4ade8099",
  },
};

const fonts = `@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');`;
