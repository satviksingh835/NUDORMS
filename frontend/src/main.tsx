import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { CapturePage } from "./pages/Capture";
import { DemoPage } from "./pages/Demo";
import { StatusPage } from "./pages/Status";
import { ViewerPage } from "./pages/Viewer";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Navigate to="/capture" replace />} />
      <Route path="/capture" element={<CapturePage />} />
      <Route path="/scans/:id" element={<StatusPage />} />
      <Route path="/scans/:id/view" element={<ViewerPage />} />
      <Route path="/demo" element={<DemoPage />} />
    </Routes>
  </BrowserRouter>,
);
