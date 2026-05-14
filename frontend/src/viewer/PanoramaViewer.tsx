import { useEffect, useRef, useState } from "react";
import "@photo-sphere-viewer/core/index.css";

export interface GraphNode {
  id: string;
  pano_key: string;
  position: [number, number, number];
}

export interface GraphEdge {
  from: string;
  to: string;
  azimuth_from: number;   // degrees [0,360), where 0 = pano image center
  azimuth_to: number;
  distance_m: number;
}

export interface GraphJSON {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface Props {
  graph: GraphJSON;
  panoUrls: Record<string, string>;
  initialNodeId?: string;
}

export function PanoramaViewer({ graph, panoUrls, initialNodeId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const markersPluginRef = useRef<any>(null);
  const [currentNodeId, setCurrentNodeId] = useState<string>(
    initialNodeId ?? graph.nodes[0]?.id ?? ""
  );

  // Navigate to a node: load its pano and set markers for its outgoing edges
  const navigateTo = (nodeId: string, viewer: any, markersPlugin: any) => {
    const url = panoUrls[nodeId];
    if (!url) return;
    const outEdges = graph.edges.filter((e) => e.from === nodeId);

    viewer.setPanorama(url).then(() => {
      markersPlugin.clearMarkers();
      outEdges.forEach((edge) => {
        const yaw = ((edge.azimuth_from % 360) * Math.PI) / 180;
        markersPlugin.addMarker({
          id: `arrow-${edge.to}`,
          position: { yaw, pitch: -0.5 },  // ~-30° pitch, near the floor horizon
          html: arrowHtml(edge.distance_m),
          size: { width: 80, height: 80 },
          style: { cursor: "pointer" },
          data: { targetNodeId: edge.to },
        });
      });
      setCurrentNodeId(nodeId);
    });
  };

  useEffect(() => {
    if (!containerRef.current || graph.nodes.length === 0) return;

    let viewer: any = null;
    let cancelled = false;

    (async () => {
      const [{ Viewer }, { MarkersPlugin }] = await Promise.all([
        import("@photo-sphere-viewer/core"),
        import("@photo-sphere-viewer/markers-plugin"),
      ]);
      if (cancelled) return;

      const startNodeId = initialNodeId ?? graph.nodes[0].id;
      const startUrl = panoUrls[startNodeId] ?? "";

      viewer = new Viewer({
        container: containerRef.current!,
        panorama: startUrl,
        plugins: [[MarkersPlugin, {}]],
        defaultYaw: 0,
        defaultPitch: 0,
        touchmoveTwoFingers: false,
        navbar: false,
      });
      viewerRef.current = viewer;

      const markersPlugin: any = viewer.getPlugin(MarkersPlugin);
      markersPluginRef.current = markersPlugin;

      // Set initial markers
      const outEdges = graph.edges.filter((e) => e.from === startNodeId);
      viewer.addEventListener("ready", () => {
        outEdges.forEach((edge) => {
          const yaw = ((edge.azimuth_from % 360) * Math.PI) / 180;
          markersPlugin.addMarker({
            id: `arrow-${edge.to}`,
            position: { yaw, pitch: -0.5 },
            html: arrowHtml(edge.distance_m),
            size: { width: 80, height: 80 },
            style: { cursor: "pointer" },
            data: { targetNodeId: edge.to },
          });
        });
      }, { once: true });

      // Click handler for arrow markers
      markersPlugin.addEventListener("select-marker", ({ marker }: any) => {
        const targetId = marker.data?.targetNodeId;
        if (targetId) {
          navigateTo(targetId, viewer, markersPlugin);
        }
      });
    })();

    return () => {
      cancelled = true;
      viewerRef.current?.destroy?.();
      viewerRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Preload neighbor images after node changes
  useEffect(() => {
    if (!currentNodeId) return;
    const neighbors = graph.edges
      .filter((e) => e.from === currentNodeId)
      .map((e) => e.to);
    neighbors.forEach((nid) => {
      const url = panoUrls[nid];
      if (url) {
        const img = new Image();
        img.src = url;
      }
    });
  }, [currentNodeId, graph.edges, panoUrls]);

  return (
    <>
      <div ref={containerRef} style={{ position: "fixed", inset: 0 }} />
      <style>{psvStyles}</style>
    </>
  );
}

function arrowHtml(distanceM: number): string {
  const label = distanceM < 5 ? "" : `${distanceM.toFixed(0)} m`;
  return `
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;user-select:none;">
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="22" fill="rgba(255,255,255,0.18)" stroke="rgba(255,255,255,0.6)" stroke-width="2"/>
        <polyline points="16,28 24,16 32,28" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
      </svg>
      ${label ? `<span style="color:#fff;font-size:11px;font-family:monospace;background:rgba(0,0,0,0.5);padding:2px 6px;border-radius:4px;">${label}</span>` : ""}
    </div>
  `;
}

const psvStyles = `
  .psv-container { background: #0a0a0a; }
  .psv-marker { transition: opacity 0.2s; }
  .psv-marker:hover { opacity: 0.8; }
`;
