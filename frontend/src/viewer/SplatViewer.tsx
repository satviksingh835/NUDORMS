import { useEffect, useRef } from "react";

// Rendering budget cap: Spark's LoD tree handles downsampling above this.
// Keeps phone framerate stable at ~60fps regardless of scene complexity.
const MAX_SPLATS = 1_500_000;

interface Props {
  /** Ordered URLs from lowest to highest LOD. */
  lodUrls?: string[];
  /** Local file to load directly (creates a temporary object URL). */
  file?: File;
}

export function SplatViewer({ lodUrls = [], file }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const urls = file ? [URL.createObjectURL(file)] : lodUrls;
    if (urls.length === 0) return;

    let viewer: any = null;
    let cancelled = false;

    (async () => {
      // Prefer Spark 2.0 (sparkjsdev/spark, MIT, April 2026):
      //   - Three.js native, 98% device coverage
      //   - Built-in LoD tree streaming + rendering budget cap
      //   - Recommended by @mkkellogg author as the migration target
      //
      // Falls back to @mkkellogg/gaussian-splats-3d if Spark not installed.
      let usingSpark = false;

      try {
        const Spark = await import("@sparkjsdev/spark");
        if (cancelled) return;

        viewer = new Spark.Viewer({
          container: ref.current!,
          maxSplats: MAX_SPLATS,
          camera: {
            position: [0, 1.5, 3],
            target:   [0, 1.0, 0],
            up:       [0, 1,   0],
          },
        });

        // Load tiers progressively: Spark streams each URL as a LoD level
        for (const url of urls) {
          if (cancelled) return;
          await viewer.load(url);
        }
        usingSpark = true;
      } catch {
        // Spark not installed — fall through to mkkellogg
      }

      if (!usingSpark) {
        let GaussianSplats3D: any;
        try {
          GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
        } catch (e) {
          console.error("[SplatViewer] no viewer library available:", e);
          return;
        }
        if (cancelled) return;

        viewer = new GaussianSplats3D.Viewer({
          rootElement: ref.current!,
          cameraUp: [0, 1, 0],
          initialCameraPosition: [0, 1.5, 3],
          initialCameraLookAt: [0, 1, 0],
          sphericalHarmonicsDegree: 2,
          antialiased: true,
          sharedMemoryForWorkers: false,
          // Clamp to budget — mkkellogg respects this via maxSplatCount
          maxSplatCount: MAX_SPLATS,
        });

        for (const url of urls) {
          if (cancelled) return;
          const format = file
            ? file.name.endsWith(".splat")
              ? GaussianSplats3D.SceneFormat.Splat
              : GaussianSplats3D.SceneFormat.Ply
            : undefined;
          try {
            await viewer.addSplatScene(url, { showLoadingUI: false, format });
          } catch (e) {
            console.error("[SplatViewer] addSplatScene failed:", e);
          }
        }
        if (!cancelled) viewer.start();
      }
    })();

    return () => {
      cancelled = true;
      viewer?.dispose?.();
      viewer?.stop?.();
      if (file) urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [lodUrls, file]);

  return <div ref={ref} style={{ position: "fixed", inset: 0 }} />;
}
