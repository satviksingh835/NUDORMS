/**
 * PanoramaViewer — equirectangular 360° viewer built on Three.js.
 * No PSV dependency: avoids the multiple-Three.js-instance and CSS-detection bugs.
 *
 * Controls: mouse drag / touch drag to look around.
 * Arrows: HTML overlay divs projected from 3D direction vectors.
 */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export interface GraphNode {
  id: string;
  pano_key: string;
  position: [number, number, number];
}

export interface GraphEdge {
  from: string;
  to: string;
  azimuth_from: number;   // degrees [0,360) clockwise from pano center
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

interface ArrowOverlay {
  edgeTo: string;
  x: number;   // CSS % from left
  y: number;   // CSS % from top
  dist: number;
  visible: boolean;
}

export function PanoramaViewer({ graph, panoUrls, initialNodeId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    camera: THREE.PerspectiveCamera;
    scene: THREE.Scene;
    mesh: THREE.Mesh;
    loader: THREE.TextureLoader;
    animId: number;
    dragging: boolean;
    lastX: number;
    lastY: number;
    lon: number;   // horizontal look angle, degrees
    lat: number;   // vertical look angle, degrees
  } | null>(null);

  const [currentNodeId, setCurrentNodeId] = useState(
    initialNodeId ?? graph.nodes[0]?.id ?? ""
  );
  const [arrows, setArrows] = useState<ArrowOverlay[]>([]);

  // Compute 2D arrow positions from 3D directions
  const updateArrows = (nodeId: string, lon: number, lat: number) => {
    const outEdges = graph.edges.filter((e) => e.from === nodeId);
    if (!canvasRef.current) return;
    const s = stateRef.current;
    if (!s) return;

    // Sync camera direction for projection
    const latRad = THREE.MathUtils.degToRad(Math.max(-85, Math.min(85, lat)));
    const lonRad = THREE.MathUtils.degToRad(lon);
    const target = new THREE.Vector3(
      Math.cos(latRad) * Math.sin(lonRad),
      Math.sin(latRad),
      Math.cos(latRad) * Math.cos(lonRad),
    );
    s.camera.lookAt(target);
    s.camera.updateMatrixWorld();

    const projected = outEdges.map((edge) => {
      // azimuth_from: 0° = forward (pano center), clockwise positive
      const az = THREE.MathUtils.degToRad(edge.azimuth_from);
      // Arrow sits at pitch -25° (near floor horizon)
      const pitch = THREE.MathUtils.degToRad(-25);
      // World direction of this arrow (arrow azimuth is relative to pano center = lon=0)
      const arrowLonRad = THREE.MathUtils.degToRad(lon) - az; // subtract because clockwise
      const dir = new THREE.Vector3(
        Math.cos(pitch) * Math.sin(arrowLonRad),
        Math.sin(pitch),
        Math.cos(pitch) * Math.cos(arrowLonRad),
      );

      // Project world point (camera-relative) to NDC
      const worldPt = dir.clone().multiplyScalar(100);
      const ndc = worldPt.project(s.camera);
      const x = ((ndc.x + 1) / 2) * 100;
      const y = ((1 - ndc.y) / 2) * 100;
      const visible = ndc.z < 1 && x > 5 && x < 95 && y > 5 && y < 95;

      return { edgeTo: edge.to, x, y, dist: edge.distance_m, visible };
    });

    setArrows(projected);
  };

  const loadPano = (nodeId: string) => {
    const s = stateRef.current;
    if (!s) return;
    const url = panoUrls[nodeId];
    if (!url) { console.error("[Pano] no url for node", nodeId); return; }
    console.log("[Pano] loading", nodeId, url.slice(0, 60));
    s.loader.load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        (s.mesh.material as THREE.MeshBasicMaterial).map = tex;
        (s.mesh.material as THREE.MeshBasicMaterial).needsUpdate = true;
        console.log("[Pano] texture loaded OK for", nodeId);
        setCurrentNodeId(nodeId);
        updateArrows(nodeId, s.lon, s.lat);
      },
      undefined,
      (err) => console.error("[Pano] texture load error:", err),
    );
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    console.log("[Pano] init Three.js renderer");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 0);

    const scene = new THREE.Scene();
    const geo = new THREE.SphereGeometry(500, 60, 40);
    geo.scale(-1, 1, 1);   // flip normals to render inside
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);

    const loader = new THREE.TextureLoader();

    let animId = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let lon = 0;
    let lat = 0;

    stateRef.current = { renderer, camera, scene, mesh, loader, animId, dragging, lastX, lastY, lon, lat };
    const s = stateRef.current;

    const animate = () => {
      s.animId = requestAnimationFrame(animate);
      const latRad = THREE.MathUtils.degToRad(Math.max(-85, Math.min(85, s.lat)));
      const lonRad = THREE.MathUtils.degToRad(s.lon);
      camera.lookAt(
        Math.cos(latRad) * Math.sin(lonRad),
        Math.sin(latRad),
        Math.cos(latRad) * Math.cos(lonRad),
      );
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      updateArrows(s.lat === 0 && s.lon === 0 ? (initialNodeId ?? graph.nodes[0]?.id ?? "") : currentNodeId, s.lon, s.lat);
    };
    window.addEventListener("resize", onResize);

    // Mouse drag
    const onMouseDown = (e: MouseEvent) => { s.dragging = true; s.lastX = e.clientX; s.lastY = e.clientY; };
    const onMouseMove = (e: MouseEvent) => {
      if (!s.dragging) return;
      s.lon -= (e.clientX - s.lastX) * 0.25;
      s.lat -= (e.clientY - s.lastY) * 0.25;
      s.lastX = e.clientX; s.lastY = e.clientY;
    };
    const onMouseUp = () => { s.dragging = false; };

    // Touch drag
    const onTouchStart = (e: TouchEvent) => { s.dragging = true; s.lastX = e.touches[0].clientX; s.lastY = e.touches[0].clientY; };
    const onTouchMove = (e: TouchEvent) => {
      if (!s.dragging) return;
      s.lon -= (e.touches[0].clientX - s.lastX) * 0.25;
      s.lat -= (e.touches[0].clientY - s.lastY) * 0.25;
      s.lastX = e.touches[0].clientX; s.lastY = e.touches[0].clientY;
    };
    const onTouchEnd = () => { s.dragging = false; };

    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("touchend", onTouchEnd);

    // Load initial panorama
    const startId = initialNodeId ?? graph.nodes[0]?.id ?? "";
    loadPano(startId);

    return () => {
      cancelAnimationFrame(s.animId);
      renderer.dispose();
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
      stateRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recompute arrow positions on every animation frame while dragging
  useEffect(() => {
    if (!currentNodeId) return;
    let id: number;
    const tick = () => {
      const s = stateRef.current;
      if (s) updateArrows(currentNodeId, s.lon, s.lat);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentNodeId]);

  // Preload neighbors
  useEffect(() => {
    graph.edges.filter((e) => e.from === currentNodeId).forEach((e) => {
      const url = panoUrls[e.to];
      if (url) { const img = new Image(); img.src = url; }
    });
  }, [currentNodeId, graph.edges, panoUrls]);

  const handleArrowClick = (toNodeId: string) => {
    const s = stateRef.current;
    if (s) { s.lon = 0; s.lat = 0; }
    loadPano(toNodeId);
  };

  return (
    <div style={{ position: "fixed", inset: 0, cursor: "grab" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%" }}
      />
      {/* Arrow overlays */}
      {arrows.filter((a) => a.visible).map((a) => (
        <button
          key={a.edgeTo}
          onClick={() => handleArrowClick(a.edgeTo)}
          style={{
            position: "absolute",
            left: `${a.x}%`,
            top: `${a.y}%`,
            transform: "translate(-50%, -50%)",
            background: "rgba(255,255,255,0.15)",
            border: "2px solid rgba(255,255,255,0.6)",
            borderRadius: "50%",
            width: 64,
            height: 64,
            cursor: "pointer",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 2,
            backdropFilter: "blur(4px)",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.3)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <polyline points="6,15 12,7 18,15" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          {a.dist >= 3 && (
            <span style={{ color: "#fff", fontSize: 9, fontFamily: "monospace", lineHeight: 1 }}>
              {a.dist.toFixed(0)}m
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
