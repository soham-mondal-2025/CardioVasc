// CardioVision 3D - Interactive Cardiac Viewer & Multimodal Risk Pipeline
let scene, camera, renderer, controls;
let heartModel = null;
let vesselMeshes = {};
let raycaster, mouse;
let isBeating = true;
let isXray = false;
let clock = new THREE.Clock();
let shapChartInstance = null;

let currentPredictions = {
    overall_cad: { probability: 88.4, is_cad: true },
    vessels: {
        LAD: { probability: 82.1, risk_color: "#ef4444" },
        LCX: { probability: 54.2, risk_color: "#f59e0b" },
        RCA: { probability: 28.7, risk_color: "#10b981" }
    }
};

let samplePatientsList = [];

// Initialize Three.js 3D Canvas
function initThree() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Scene & Camera
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 7.5);

    // 2. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    // 3. Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 3;
    controls.maxDistance = 15;
    controls.autoRotate = false;

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.6); // Cool rim light
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    const keyLight = new THREE.PointLight(0xf43f5e, 0.8, 10); // Warm cardiac glow
    keyLight.position.set(0, 2, 4);
    scene.add(keyLight);

    // 5. Raycasting for Interaction
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // 6. Load Anatomical Heart GLTF/GLB Model
    const loader = new THREE.GLTFLoader();
    loader.load(
        '/static/models/heart.glb',
        (gltf) => {
            heartModel = gltf.scene;

            // Center and scale heart mesh
            const box = new THREE.Box3().setFromObject(heartModel);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 3.6 / maxDim;
            heartModel.scale.set(scale, scale, scale);
            heartModel.position.sub(center.multiplyScalar(scale));

            // Customize heart myocardium shader material
            heartModel.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshStandardMaterial({
                        color: 0x991b1b, // Deep cardiac muscle crimson
                        roughness: 0.35,
                        metalness: 0.1,
                        clearcoat: 0.3,
                        clearcoatRoughness: 0.2
                    });
                }
            });

            scene.add(heartModel);

            // Construct 3D Coronary Artery Vessels (LAD, LCX, RCA)
            buildCoronaryVessels();

            // Hide loader
            const loaderEl = document.getElementById('loader');
            if (loaderEl) loaderEl.style.display = 'none';

            // Initial vessel coloring
            updateVesselColors(currentPredictions.vessels);
        },
        undefined,
        (error) => {
            console.error('Error loading heart.glb:', error);
            // Fallback: procedural 3D myocardium if glb fails
            buildFallbackHeart();
            buildCoronaryVessels();
            const loaderEl = document.getElementById('loader');
            if (loaderEl) loaderEl.style.display = 'none';
        }
    );

    // 7. Event Listeners
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('click', onCanvasClick);

    // Animation Loop
    animate();
}

// Fallback procedural heart geometry in case mesh loading network issue
function buildFallbackHeart() {
    const group = new THREE.Group();
    const geo = new THREE.SphereGeometry(1.6, 32, 32);
    // Deform into heart shape
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        let x = pos.getX(i);
        let y = pos.getY(i);
        let z = pos.getZ(i);
        if (y < 0) {
            x *= (1 + y * 0.3);
            z *= (1 + y * 0.3);
        }
        pos.setXYZ(i, x, y * 1.15, z);
    }
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
        color: 0x991b1b,
        roughness: 0.3,
        metalness: 0.1
    });
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);
    heartModel = group;
    scene.add(heartModel);
}

// Construct Anatomical Coronary Artery 3D Curves
function buildCoronaryVessels() {
    // 1. LAD (Left Anterior Descending): Runs down front interventricular groove to apex
    const ladCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-0.1, 1.2, 0.9),
        new THREE.Vector3(-0.3, 0.7, 1.25),
        new THREE.Vector3(-0.4, 0.1, 1.35),
        new THREE.Vector3(-0.35, -0.6, 1.2),
        new THREE.Vector3(-0.15, -1.2, 0.8),
        new THREE.Vector3(0.0, -1.5, 0.4) // Apex
    ]);

    // 2. LCX (Left Circumflex): Wraps around lateral side and back of left ventricle
    const lcxCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-0.1, 1.2, 0.9),
        new THREE.Vector3(-0.7, 1.0, 0.8),
        new THREE.Vector3(-1.3, 0.6, 0.4),
        new THREE.Vector3(-1.45, 0.0, -0.3),
        new THREE.Vector3(-1.2, -0.6, -0.7),
        new THREE.Vector3(-0.7, -1.0, -0.6)
    ]);

    // 3. RCA (Right Coronary Artery): Runs along right atrioventricular sulcus to inferior wall
    const rcaCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0.2, 1.2, 0.9),
        new THREE.Vector3(0.8, 0.9, 0.9),
        new THREE.Vector3(1.3, 0.4, 0.6),
        new THREE.Vector3(1.4, -0.2, 0.1),
        new THREE.Vector3(1.1, -0.8, -0.4),
        new THREE.Vector3(0.4, -1.2, -0.6) // Inferior / posterior descending
    ]);

    const vessels = [
        { id: 'LAD', curve: ladCurve, radius: 0.075 },
        { id: 'LCX', curve: lcxCurve, radius: 0.065 },
        { id: 'RCA', curve: rcaCurve, radius: 0.075 }
    ];

    vessels.forEach((v) => {
        const tubeGeo = new THREE.TubeGeometry(v.curve, 64, v.radius, 12, false);
        const tubeMat = new THREE.MeshStandardMaterial({
            color: 0x10b981,
            roughness: 0.2,
            metalness: 0.3,
            emissive: 0x10b981,
            emissiveIntensity: 0.4
        });
        const mesh = new THREE.Mesh(tubeGeo, tubeMat);
        mesh.userData = {
            isVessel: true,
            vesselId: v.id,
            name: v.id === 'LAD' ? 'Left Anterior Descending' : (v.id === 'LCX' ? 'Left Circumflex' : 'Right Coronary Artery')
        };
        scene.add(mesh);
        vesselMeshes[v.id] = mesh;

        // Also add glowing pulsating marker beads along the vessel
        const sphereGeo = new THREE.SphereGeometry(0.12, 16, 16);
        const sphereMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
        const marker = new THREE.Mesh(sphereGeo, sphereMat);
        const pt = v.curve.getPoint(0.5);
        marker.position.copy(pt);
        marker.userData = { isVessel: true, vesselId: v.id };
        scene.add(marker);
        vesselMeshes[v.id + '_marker'] = marker;
    });
}

// Update 3D vessel colors based on prediction probabilities
function updateVesselColors(vesselsData) {
    if (!vesselsData) return;

    ['LAD', 'LCX', 'RCA'].forEach((vid) => {
        const vInfo = vesselsData[vid];
        if (!vInfo) return;

        const colorHex = parseInt(vInfo.risk_color.replace('#', '0x'));
        const mesh = vesselMeshes[vid];
        const marker = vesselMeshes[vid + '_marker'];

        if (mesh && mesh.material) {
            mesh.material.color.setHex(colorHex);
            mesh.material.emissive.setHex(colorHex);
            mesh.material.emissiveIntensity = vInfo.probability > 60 ? 0.7 : 0.3;
        }
        if (marker && marker.material) {
            marker.material.color.setHex(colorHex);
        }

        // Update UI indicator dots
        const dot = document.getElementById('dot' + vid);
        const badgeDot = document.getElementById('badgeDot' + vid);
        if (dot) dot.style.backgroundColor = vInfo.risk_color;
        if (badgeDot) badgeDot.style.backgroundColor = vInfo.risk_color;
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    const time = clock.getElapsedTime();

    // Subtle cardiac cycle contraction (beating pulse)
    if (heartModel && isBeating) {
        const pulse = 1.0 + Math.sin(time * 4.5) * 0.025;
        heartModel.scale.set(pulse * 1.0, pulse * 1.0, pulse * 1.0);
    }

    // Severe stenosis pulse danger glow
    ['LAD', 'LCX', 'RCA'].forEach((vid) => {
        const marker = vesselMeshes[vid + '_marker'];
        const vInfo = currentPredictions.vessels[vid];
        if (marker && vInfo && vInfo.probability > 60) {
            const glow = 1.0 + Math.sin(time * 7) * 0.3;
            marker.scale.set(glow, glow, glow);
        }
    });

    controls.update();
    renderer.render(scene, camera);
}

// Window resize
function onWindowResize() {
    const container = document.getElementById('three-container');
    if (!container || !renderer || !camera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

// Mouse move raycasting for tooltips
function onMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    const tooltip = document.getElementById('tooltip3d');
    let hitVessel = null;

    for (let hit of intersects) {
        if (hit.object.userData && hit.object.userData.isVessel) {
            hitVessel = hit.object.userData.vesselId;
            break;
        }
    }

    if (hitVessel) {
        const vInfo = currentPredictions.vessels[hitVessel];
        document.getElementById('ttTitle').innerText = `${hitVessel}: ${vInfo.name}`;
        document.getElementById('ttDetails').innerText = `Territory: ${vInfo.territory}`;
        document.getElementById('ttRisk').innerHTML = `
            <span>Predicted Stenosis:</span>
            <span class="font-mono font-bold" style="color: ${vInfo.risk_color};">${vInfo.probability}% (${vInfo.stenosis ? 'Stenotic &ge;50%' : 'Normal'})</span>
        `;
        tooltip.style.left = `${event.clientX - rect.left + 15}px`;
        tooltip.style.top = `${event.clientY - rect.top + 15}px`;
        tooltip.classList.remove('hidden');
    } else {
        tooltip.classList.add('hidden');
    }
}

function onCanvasClick(event) {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);
    for (let hit of intersects) {
        if (hit.object.userData && hit.object.userData.isVessel) {
            focusVessel(hit.object.userData.vesselId);
            break;
        }
    }
}

// Smooth camera focus on specific coronary vessels
function focusVessel(vesselId) {
    const targets = {
        'LAD': { cam: [0, 0, 6.5], look: [0, 0, 0] },     // Front view
        'LCX': { cam: [-5, 1.5, 3.5], look: [-0.5, 0, 0] }, // Left Lateral
        'RCA': { cam: [5, 1.5, 3.5], look: [0.5, 0, 0] }   // Right Lateral
    };

    const t = targets[vesselId];
    if (t) {
        camera.position.set(...t.cam);
        controls.target.set(...t.look);
        controls.update();
    }
}

// -------------------------------------------------------------
// Clinical Dashboard & API Integration
// -------------------------------------------------------------

async function loadSamplePatients() {
    try {
        const res = await fetch('/api/sample-patients');
        samplePatientsList = await res.json();
        const sel = document.getElementById('presetSelect');
        sel.innerHTML = '';
        samplePatientsList.forEach((p) => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.innerText = p.name;
            sel.appendChild(opt);
        });

        // Set initial preset
        selectPreset('patient-1');
    } catch (e) {
        console.error('Error fetching sample patients:', e);
    }
}

function selectPreset(pid) {
    const p = samplePatientsList.find(x => x.id === pid);
    if (!p) return;

    document.getElementById('presetDesc').innerText = p.description;

    // Populate inputs
    for (const [key, val] of Object.entries(p.data)) {
        const el = document.getElementById('inp_' + key.replace(/ /g, '_').replace(/-/g, '_'));
        if (el) el.value = val;
    }

    // Execute prediction immediately
    executePrediction();
}

function collectFormData() {
    return {
        "Age": parseFloat(document.getElementById('inp_Age').value),
        "Sex": parseInt(document.getElementById('inp_Sex').value),
        "BP": parseFloat(document.getElementById('inp_BP').value),
        "PR": parseFloat(document.getElementById('inp_PR').value),
        "BMI": parseFloat(document.getElementById('inp_BMI').value),
        "Typical Chest Pain": parseInt(document.getElementById('inp_Typical_Chest_Pain').value),
        "Region RWMA": parseInt(document.getElementById('inp_Region_RWMA').value),
        "EF-TTE": parseFloat(document.getElementById('inp_EF_TTE').value),
        "Tinversion": parseInt(document.getElementById('inp_Tinversion').value),
        "St Depression": parseInt(document.getElementById('inp_St_Depression').value),
        "St Elevation": parseInt(document.getElementById('inp_St_Elevation').value),
        "Poor R Progression": parseInt(document.getElementById('inp_Poor_R_Progression').value),
        "TG": parseFloat(document.getElementById('inp_TG').value),
        "LDL": parseFloat(document.getElementById('inp_LDL').value),
        "HDL": parseFloat(document.getElementById('inp_HDL').value),
        "FBS": parseFloat(document.getElementById('inp_FBS').value),
        "HTN": parseInt(document.getElementById('inp_HTN').value),
        "DM": parseInt(document.getElementById('inp_DM').value),
        "Q Wave": 0,
        "Current Smoker": 0,
        "Weight": 72,
        "Length": 170,
        "Lymph": 28,
        "WBC": 7000
    };
}

async function executePrediction() {
    const btn = document.getElementById('btnPredict');
    const origHTML = btn.innerHTML;
    btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> Analyzing...`;
    lucide.createIcons();

    try {
        const payload = { patient: collectFormData() };
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        currentPredictions = data;

        // 1. Update Overall CAD Card
        const cad = data.overall_cad;
        document.getElementById('txtCadProb').innerText = `${cad.probability}%`;
        document.getElementById('barCadProb').style.width = `${cad.probability}%`;
        const badge = document.getElementById('badgeCadStatus');
        badge.innerText = cad.risk_category.toUpperCase();

        if (cad.probability >= 60) {
            badge.className = "px-2 py-0.5 text-xs font-bold rounded bg-rose-500/20 text-rose-400 border border-rose-500/30";
            document.getElementById('txtCadSummary').innerText = "High-grade coronary stenosis likely. Recommend urgent diagnostic catheterization.";
        } else if (cad.probability >= 35) {
            badge.className = "px-2 py-0.5 text-xs font-bold rounded bg-amber-500/20 text-amber-400 border border-amber-500/30";
            document.getElementById('txtCadSummary').innerText = "Moderate cardiovascular risk. Follow-up stress imaging and lipid optimization suggested.";
        } else {
            badge.className = "px-2 py-0.5 text-xs font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
            document.getElementById('txtCadSummary').innerText = "Low risk profile. Physiological metrics within safe baseline parameters.";
        }

        // 2. Update Vessels Cards
        const vessels = data.vessels;
        document.getElementById('txtLadProb').innerText = `${vessels.LAD.probability}%`;
        document.getElementById('txtLadProb').style.color = vessels.LAD.risk_color;

        document.getElementById('txtLcxProb').innerText = `${vessels.LCX.probability}%`;
        document.getElementById('txtLcxProb').style.color = vessels.LCX.risk_color;

        document.getElementById('txtRcaProb').innerText = `${vessels.RCA.probability}%`;
        document.getElementById('txtRcaProb').style.color = vessels.RCA.risk_color;

        // 3. Update 3D Vessel Mesh Colors
        updateVesselColors(vessels);

        // 4. Update SHAP Chart
        renderShapChart(data.shap_explanation);

    } catch (err) {
        console.error('Prediction failed:', err);
    } finally {
        btn.innerHTML = origHTML;
        lucide.createIcons();
    }
}

// Render SHAP TreeExplainer Waterfall Bar Chart
function renderShapChart(shapList) {
    if (!shapList || !shapList.length) return;

    const ctx = document.getElementById('shapChart').getContext('2d');
    const labels = shapList.map(s => s.feature);
    const dataVals = shapList.map(s => s.importance);
    const bgColors = shapList.map(s => s.importance > 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)');

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: dataVals,
                backgroundColor: bgColors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `SHAP Impact: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(51, 65, 85, 0.4)' },
                    ticks: { color: '#94a3b8', font: { size: 9 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { size: 9 } }
                }
            }
        }
    });
}

// Model Validation Metrics Modal
async function loadMetricsModal() {
    try {
        const res = await fetch('/api/model-metrics');
        const metrics = await res.json();
        const tbody = document.getElementById('metricsTableBody');
        tbody.innerHTML = '';

        for (const [key, m] of Object.entries(metrics)) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="py-2.5 px-3 font-semibold text-rose-300">${m.label}</td>
                <td class="py-2.5 px-3 text-emerald-400 font-bold">${m.accuracy}%</td>
                <td class="py-2.5 px-3">${m.roc_auc}</td>
                <td class="py-2.5 px-3">${m.precision}</td>
                <td class="py-2.5 px-3">${m.recall}</td>
                <td class="py-2.5 px-3">${m.f1_score}</td>
            `;
            tbody.appendChild(tr);
        }
        document.getElementById('metricsModal').classList.remove('hidden');
    } catch (e) {
        console.error('Failed to load metrics:', e);
    }
}

function closeMetricsModal() {
    document.getElementById('metricsModal').classList.add('hidden');
}

// -------------------------------------------------------------
// DOM Ready Setup
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initThree();
    loadSamplePatients();

    document.getElementById('presetSelect').addEventListener('change', (e) => {
        selectPreset(e.target.value);
    });

    document.getElementById('btnPredict').addEventListener('click', executePrediction);

    document.getElementById('btnResetCam').addEventListener('click', () => {
        camera.position.set(0, 0, 7.5);
        controls.target.set(0, 0, 0);
        controls.update();
    });

    document.getElementById('btnBeatingToggle').addEventListener('click', () => {
        isBeating = !isBeating;
        if (!isBeating && heartModel) heartModel.scale.set(1, 1, 1);
    });

    document.getElementById('btnXrayToggle').addEventListener('click', () => {
        isXray = !isXray;
        if (heartModel) {
            heartModel.traverse((child) => {
                if (child.isMesh) {
                    child.material.transparent = isXray;
                    child.material.opacity = isXray ? 0.35 : 1.0;
                }
            });
        }
    });

    document.getElementById('btnMetricsModal').addEventListener('click', loadMetricsModal);
});
