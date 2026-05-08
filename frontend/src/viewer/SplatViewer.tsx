import { useEffect, useRef } from "react";

interface Props {
  /** Ordered URLs from lowest to highest LOD. */
  lodUrls: string[];
}

/**
 * Splat viewer using @mkkellogg/gaussian-splats-3d. Loads the lowest LOD
 * first so the user sees something within ~500ms, then progressively swaps
 * in the higher tiers. WebGPU path with WebGL2 fallback is handled by the
 * library based on browser support.
 */
export function SplatViewer({ lodUrls }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || lodUrls.length === 0) return;
    let viewer: any = null;
    let cancelled = false;

    (async () => {
      const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
      viewer = new GaussianSplats3D.Viewer({
        rootElement: ref.current!,
        cameraUp: [0, 1, 0],
        initialCameraPosition: [0, 1.5, 3],
        initialCameraLookAt: [0, 1, 0],
        sphericalHarmonicsDegree: 2,
        antialiased: true,            // Mip-Splatting AA
        sharedMemoryForWorkers: true,
      });

      for (const url of lodUrls) {
        if (cancelled) return;
        await viewer.addSplatScene(url, { showLoadingUI: false });
      }
      if (!cancelled) viewer.start();
    })();

    return () => {
      cancelled = true;
      viewer?.dispose?.();
    };
  }, [lodUrls]);

  return <div ref={ref} style={{ position: "fixed", inset: 0 }} />;
}
