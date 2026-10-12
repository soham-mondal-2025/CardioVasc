const fs = require("fs");
const path = require("path");
const THREE = require("three");
const { OBJLoader } = require("three-stdlib");
const { GLTFExporter } = require("three-stdlib");

const inputDir = path.resolve(
  __dirname,
  "../public/models/bodyparts3d-test"
);

const outputDir = path.join(inputDir, "glb");

fs.mkdirSync(outputDir, { recursive: true });

const ids = [
  "FJ2723",
  "FJ2737",
  "FJ2649",
  "FJ2650",
  "FJ2651",
  "FJ2652",
  "FJ2653",
  "FJ2654",
  "FJ2631"
];

const loader = new OBJLoader();
const exporter = new GLTFExporter();

async function convert(id) {
  const objPath = path.join(inputDir, `${id}.obj`);
  const glbPath = path.join(outputDir, `${id}.glb`);

  if (!fs.existsSync(objPath)) {
    throw new Error(`Missing OBJ file: ${objPath}`);
  }

  const objText = fs.readFileSync(objPath, "utf8");
  const object = loader.parse(objText);

  object.name = id;

  object.traverse((child) => {
    if (child.isMesh) {
      child.name = id;
      child.material = new THREE.MeshStandardMaterial({
        color: 0xc95b59,
        roughness: 0.8,
        metalness: 0
      });

      child.geometry.computeVertexNormals();
    }
  });

  const glb = await new Promise((resolve, reject) => {
    exporter.parse(
      object,
      resolve,
      reject,
      { binary: true }
    );
  });

  fs.writeFileSync(glbPath, Buffer.from(glb));
  console.log(`Converted ${id}: ${fs.statSync(glbPath).size} bytes`);
}

(async () => {
  for (const id of ids) {
    await convert(id);
  }

  console.log("Conversion finished.");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
