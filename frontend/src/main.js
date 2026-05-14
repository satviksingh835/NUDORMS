import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { CapturePage } from "./pages/Capture";
import { DemoPage } from "./pages/Demo";
import { StatusPage } from "./pages/Status";
import { ViewerPage } from "./pages/Viewer";
ReactDOM.createRoot(document.getElementById("root")).render(_jsx(BrowserRouter, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Navigate, { to: "/capture", replace: true }) }), _jsx(Route, { path: "/capture", element: _jsx(CapturePage, {}) }), _jsx(Route, { path: "/scans/:id", element: _jsx(StatusPage, {}) }), _jsx(Route, { path: "/scans/:id/view", element: _jsx(ViewerPage, {}) }), _jsx(Route, { path: "/demo", element: _jsx(DemoPage, {}) })] }) }));
