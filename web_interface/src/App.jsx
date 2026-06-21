import React, { useEffect } from "react";
import "./index.css";
import htmlContent from "./legacy_body.html?raw";

export default function App() {
  useEffect(() => {
    // Only append if it doesn't exist
    if (!document.getElementById("legacy-script")) {
      const script = document.createElement("script");
      script.id = "legacy-script";
      script.src = "/legacy_script.js";
      // We append to the body or head
      document.body.appendChild(script);
    }
  }, []);

  return <div dangerouslySetInnerHTML={{ __html: htmlContent }} />;
}
