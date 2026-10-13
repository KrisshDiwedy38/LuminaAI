import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

document.documentElement.dataset.theme = new URLSearchParams(location.search).get("theme") === "dark" ? "dark" : "light";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
