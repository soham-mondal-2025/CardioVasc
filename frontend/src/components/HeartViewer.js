
import React, { Suspense, useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import {
  Bounds,
  Center,
  Html,
  OrbitControls,
  useGLTF,
} from "@react-three/drei";
import "./HeartViewer.css";

function HeartModel({ onSelect, selectedName, diseaseMode }) {
  const { scene } = useGLTF("/models/heart.glb");

  const model = useMemo(() => {
    const copy = scene.clone(true);

    copy.traverse((object) => {
      if (!object.isMesh) return;

      object.material = Array.isArray(object.material)
        ? object.material.map((material) => material.clone())
        : object.material.clone();

      object.castShadow = true;
      object.receiveShadow = true;
    });

    return copy;
  }, [scene]);

  useEffect(() => {
    model.traverse((object) => {
      if (!object.isMesh) return;

      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];

      const name = object.name || "";
      const selected = name === selectedName;
      const coronaryName =
        diseaseMode === "coronary" &&
        /(coronary|artery|arteries|vessel)/i.test(name);

      materials.forEach((material) => {
        if (!material.emissive) return;

        if (selected) {
          material.emissive.set("#00d9ff");
          material.emissiveIntensity = 0.7;
        } else if (coronaryName) {
          material.emissive.set("#ff3030");
          material.emissiveIntensity = 0.35;
        } else {
          material.emissive.set("#000000");
          material.emissiveIntensity = 0;
        }
      });
    });
  }, [model, selectedName, diseaseMode]);

  return (
    <primitive
      object={model}
      onClick={(event) => {
        event.stopPropagation();

        if (event.object?.isMesh) {
          onSelect(event.object.name || "Unnamed mesh");
        }
      }}
    />
  );
}

function LoadingMessage() {
  return (
    <Html center>
      <div className="heart-loading">Loading heart model…</div>
    </Html>
  );
}

export default function HeartViewer({
  predictionData = null,
  loading = false,
}) {
  const [selectedName, setSelectedName] = useState("");
  const [diseaseMode, setDiseaseMode] = useState("none");

  return (
    <section className="heart-viewer">
      <div className="heart-viewer-header">
        <div>
          <h2>Interactive Heart Anatomy</h2>
          <p>Drag to rotate · Scroll to zoom · Click a mesh to inspect</p>
        </div>

        <label className="heart-mode-control">
          <span>Visualization</span>
          <select
            value={diseaseMode}
            onChange={(event) => setDiseaseMode(event.target.value)}
          >
            <option value="none">Normal view</option>
            <option value="coronary">Coronary vessel name overlay</option>
          </select>
        </label>
      </div>

      <div className="heart-canvas">
        <Canvas
          camera={{ position: [0, 0, 5], fov: 40, near: 0.01, far: 1000 }}
          dpr={[1, 1.5]}
          onPointerMissed={() => setSelectedName("")}
        >
          <color attach="background" args={["#101723"]} />
          <ambientLight intensity={1.8} />
          <directionalLight position={[4, 5, 6]} intensity={2.2} />
          <directionalLight position={[-4, 2, -3]} intensity={1.2} />

          <Suspense fallback={<LoadingMessage />}>
            <Bounds fit clip observe margin={1.25}>
              <Center>
                <HeartModel
                  onSelect={setSelectedName}
                  selectedName={selectedName}
                  diseaseMode={diseaseMode}
                />
              </Center>
            </Bounds>
          </Suspense>

          <OrbitControls
            makeDefault
            enableRotate
            enableZoom
            enablePan
            minDistance={0.15}
            maxDistance={30}
            zoomSpeed={1.2}
          />
        </Canvas>
      </div>

      <div className="heart-inspector">
        <div>
          <h3>Selected model part</h3>
          <p>{selectedName || "Click a visible part to inspect its mesh name."}</p>
        </div>

        <div>
          <h3>Prediction status</h3>
          {loading ? (
            <p>Calculating prediction…</p>
          ) : predictionData ? (
            <pre className="prediction-output">
              {JSON.stringify(predictionData, null, 2)}
            </pre>
          ) : (
            <p>Run a demo prediction using the clinical feature form.</p>
          )}
        </div>
      </div>

      <p className="heart-disclaimer">
        Educational visualization only. Highlighting by mesh name is not
        disease detection. A prediction does not confirm a diagnosis or
        identify a blockage location.
      </p>
    </section>
  );
}

useGLTF.preload("/models/heart.glb");
