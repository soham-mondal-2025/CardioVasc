
import React, { Suspense, useMemo, useState } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { GLTFLoader } from "three-stdlib";
import * as THREE from "three";

const arteries = [
  {
    id: "FJ2723",
    label: "Right coronary artery trunk",
    color: "#d94f50",
  },
  {
    id: "FJ2737",
    label: "Left coronary artery trunk",
    color: "#f08a5d",
  },
  {
    id: "FJ2631",
    label: "Anterior interventricular branch trunk",
    color: "#e6b84a",
  },
  {
    id: "FJ2649",
    label: "Circumflex branch — piece 1",
    color: "#58a6a6",
  },
  {
    id: "FJ2650",
    label: "Circumflex branch — piece 2",
    color: "#58a6a6",
  },
  {
    id: "FJ2651",
    label: "Circumflex branch — piece 3",
    color: "#58a6a6",
  },
  {
    id: "FJ2652",
    label: "Circumflex branch — piece 4",
    color: "#58a6a6",
  },
  {
    id: "FJ2653",
    label: "Circumflex branch — piece 5",
    color: "#58a6a6",
  },
  {
    id: "FJ2654",
    label: "Circumflex branch — piece 6",
    color: "#58a6a6",
  },
];

function logBounds(label, object) {
  object.updateMatrixWorld(true);

  const bounds = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();

  bounds.getSize(size);
  bounds.getCenter(center);

  console.log(label, {
    min: bounds.min.toArray(),
    max: bounds.max.toArray(),
    center: center.toArray(),
    size: size.toArray(),
  });
}

function HeartModel() {
  const gltf = useLoader(
    GLTFLoader,
    "/models/heart.glb"
  );

  const model = useMemo(() => {
    const clone = gltf.scene.clone(true);

    // Report the original coordinates before any centering.
    logBounds("[Heart original bounds]", clone);

    return clone;
  }, [gltf.scene]);

  return <primitive object={model} />;
}

function ArteryModel({ artery, visible }) {
  const gltf = useLoader(
    GLTFLoader,
    `/models/bodyparts3d-test/glb/${artery.id}.glb`
  );

  const model = useMemo(() => {
    const clone = gltf.scene.clone(true);

    clone.traverse((object) => {
      if (!object.isMesh) return;

      object.material = new THREE.MeshStandardMaterial({
        color: artery.color,
        roughness: 0.72,
        metalness: 0,
        side: THREE.DoubleSide,
      });
    });

    logBounds(`[Artery bounds: ${artery.id}]`, clone);

    return clone;
  }, [gltf.scene, artery.color, artery.id]);

  return (
    <primitive
      object={model}
      visible={visible}
    />
  );
}

function Scene({
  visibility,
  showHeart,
  showArteries,
}) {
  return (
    <>
      <ambientLight intensity={1.6} />

      <directionalLight
        position={[200, 100, 300]}
        intensity={2.2}
      />

      <directionalLight
        position={[-150, -100, -200]}
        intensity={0.8}
      />

      {showHeart && <HeartModel />}

      {showArteries && (
        <group>
          {arteries.map((artery) => (
            <ArteryModel
              key={artery.id}
              artery={artery}
              visible={visibility[artery.id]}
            />
          ))}
        </group>
      )}

      <OrbitControls
  makeDefault
  enableDamping
  target={[25, -125, 1220]}
  minDistance={1}
  maxDistance={3000}
/>
    </>
  );
}

export default function CoronaryArteryTest() {
  const [viewMode, setViewMode] = useState("both");

  const [visibility, setVisibility] = useState(() =>
    Object.fromEntries(
      arteries.map((artery) => [artery.id, true])
    )
  );

  function toggleArtery(id) {
    setVisibility((current) => ({
      ...current,
      [id]: !current[id],
    }));
  }

  function showAllArteries() {
    setVisibility(
      Object.fromEntries(
        arteries.map((artery) => [artery.id, true])
      )
    );
  }

  return (
    <div className="coronary-test">
      <div
        className="coronary-view-modes"
        role="group"
        aria-label="Anatomy display mode"
      >
        <button
          type="button"
          aria-pressed={viewMode === "heart"}
          onClick={() => setViewMode("heart")}
        >
          Heart only
        </button>

        <button
          type="button"
          aria-pressed={viewMode === "arteries"}
          onClick={() => setViewMode("arteries")}
        >
          Arteries only
        </button>

        <button
          type="button"
          aria-pressed={viewMode === "both"}
          onClick={() => setViewMode("both")}
        >
          Heart + arteries
        </button>
      </div>

      <div className="coronary-test-toolbar">
        <p>
          Experimental anatomy preview. Drag to rotate and
          scroll to zoom. Use the controls to toggle model pieces.
        </p>

        <button
          type="button"
          onClick={showAllArteries}
        >
          Show all
        </button>
      </div>

      <div className="coronary-test-layout">
        <div className="coronary-test-canvas">
          <Canvas
            camera={{
            position: [25, -475, 1450],
            fov: 45,
            near: 0.1,
            far: 10000,
            }}
          >
            <color
              attach="background"
              args={["#f8fafb"]}
            />

            <Suspense
              fallback={
                <Html center>
                  <div className="coronary-loading">
                    Loading anatomy…
                  </div>
                </Html>
              }
            >
              <Scene
                visibility={visibility}
                showHeart={viewMode !== "arteries"}
                showArteries={viewMode !== "heart"}
              />
            </Suspense>
          </Canvas>
        </div>

        {viewMode !== "heart" && (
          <aside className="coronary-test-legend">
            <strong>Model pieces</strong>

            {arteries.map((artery) => (
              <label
                key={artery.id}
                className="coronary-test-option"
              >
                <input
                  type="checkbox"
                  checked={visibility[artery.id]}
                  onChange={() => toggleArtery(artery.id)}
                />

                <span
                  className="coronary-test-swatch"
                  style={{
                    backgroundColor: artery.color,
                  }}
                />

                <span>{artery.label}</span>
              </label>
            ))}
          </aside>
        )}
      </div>

      <p className="coronary-test-note">
        Educational software test only. These selected anatomical
        pieces are not a complete coronary artery tree or a
        patient-specific model. Their alignment with the heart
        must be verified before interpreting their anatomical position.
      </p>
    </div>
  );
}
