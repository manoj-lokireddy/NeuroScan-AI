import os
import io
import json
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image
import tensorflow as tf

# Paths & Settings
PORT = 5050
MODEL_PATH = "models/brain_tumor_best.keras"
IMG_SIZE = (224, 224)

print("Loading Brain Tumor Detection model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Pre-load sample image paths
SAMPLES = [
    {"id": "yes_1", "name": "Tumor Scan 1 (Y1.jpg)", "path": "yes/Y1.jpg", "expected": "TUMOR"},
    {"id": "yes_2", "name": "Tumor Scan 2 (Y10.jpg)", "path": "yes/Y10.jpg", "expected": "TUMOR"},
    {"id": "yes_3", "name": "Tumor Scan 3 (Y104.jpg)", "path": "yes/Y104.jpg", "expected": "TUMOR"},
    {"id": "yes_4", "name": "Tumor Scan 4 (Y188.jpg)", "path": "yes/Y188.jpg", "expected": "TUMOR"},
    {"id": "no_1", "name": "Healthy Scan 1 (1 no.jpeg)", "path": "no/1 no.jpeg", "expected": "NO TUMOR"},
    {"id": "no_2", "name": "Healthy Scan 2 (2 no.jpeg)", "path": "no/2 no.jpeg", "expected": "NO TUMOR"},
    {"id": "no_3", "name": "Healthy Scan 3 (10 no.jpg)", "path": "no/10 no.jpg", "expected": "NO TUMOR"},
    {"id": "no_4", "name": "Healthy Scan 4 (no 92.jpg)", "path": "no/no 92.jpg", "expected": "NO TUMOR"},
]

def image_to_base64(img_path):
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

def predict_pil_image(pil_img):
    rgb_img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(rgb_img, dtype=np.float32)
    # Model contains layers.Rescaling(1.0 / 255)
    img_array = np.expand_dims(img_array, axis=0)
    prob = float(model.predict(img_array, verbose=0)[0][0])
    
    if prob >= 0.5:
        result = "TUMOR DETECTED"
        confidence = prob * 100
        category = "tumor"
    else:
        result = "NO TUMOR (HEALTHY)"
        confidence = (1.0 - prob) * 100
        category = "healthy"
        
    return {
        "result": result,
        "category": category,
        "confidence": round(confidence, 2),
        "tumor_prob": round(prob * 100, 2),
        "healthy_prob": round((1.0 - prob) * 100, 2),
        "raw_score": round(prob, 4)
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Brain Tumor Detection AI - Diagnostics & Output Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 46, 0.75);
      --card-border: rgba(56, 189, 248, 0.15);
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.35);
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.4);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.4);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at 50% 0%, #172554 0%, #090d16 60%, #030712 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      border-bottom: 1px solid var(--card-border);
      background: rgba(9, 13, 22, 0.85);
      backdrop-filter: blur(16px);
      padding: 1.25rem 2.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .logo-badge {
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      width: 44px;
      height: 44px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.25rem;
      box-shadow: 0 0 20px var(--accent-glow);
    }
    .brand-title h1 {
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #ffffff, #93c5fd);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-title p {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .badge-status {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.4rem 0.9rem;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 9999px;
      font-size: 0.8rem;
      color: #34d399;
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: #34d399;
      border-radius: 50%;
      box-shadow: 0 0 10px #34d399;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    .tabs {
      display: flex;
      gap: 0.5rem;
      margin: 1.5rem 2.5rem 0.5rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 0.5rem;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.6rem 1.25rem;
      font-size: 0.95rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.05);
    }
    .tab-btn.active {
      color: var(--accent);
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }

    main {
      flex: 1;
      padding: 1.5rem 2.5rem 3rem;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    /* Scanner View Grid */
    .scanner-grid {
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      gap: 2rem;
    }
    @media (max-width: 1024px) {
      .scanner-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 1.75rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .card-header {
      margin-bottom: 1.25rem;
    }
    .card-header h2 {
      font-size: 1.15rem;
      font-weight: 700;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    /* Upload Zone */
    .upload-box {
      border: 2px dashed rgba(56, 189, 248, 0.4);
      border-radius: 14px;
      padding: 2.25rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s;
      background: rgba(56, 189, 248, 0.02);
    }
    .upload-box:hover, .upload-box.dragover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
      transform: translateY(-2px);
    }
    .upload-icon {
      font-size: 2.5rem;
      margin-bottom: 0.75rem;
    }
    .upload-box p {
      font-size: 0.9rem;
      color: var(--text-muted);
    }
    .upload-box span {
      color: var(--accent);
      font-weight: 600;
    }

    /* Samples Carousel */
    .sample-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin: 1.5rem 0 0.75rem;
    }
    .sample-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
    }
    .sample-item {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 0.5rem;
      cursor: pointer;
      transition: all 0.2s;
      text-align: center;
    }
    .sample-item:hover {
      border-color: var(--accent);
      transform: scale(1.03);
      box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2);
    }
    .sample-item.selected {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.15);
    }
    .sample-item img {
      width: 100%;
      height: 75px;
      object-fit: cover;
      border-radius: 6px;
      margin-bottom: 0.4rem;
    }
    .sample-item p {
      font-size: 0.7rem;
      font-weight: 600;
      color: #cbd5e1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Output Results Display */
    .preview-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 280px;
      background: rgba(15, 23, 42, 0.9);
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      overflow: hidden;
      position: relative;
    }
    .mri-display {
      max-width: 100%;
      max-height: 320px;
      object-fit: contain;
      border-radius: 10px;
    }
    .no-mri-text {
      color: var(--text-muted);
      font-size: 0.9rem;
      text-align: center;
    }

    /* Prediction Result Banner */
    .result-banner {
      margin-top: 1.5rem;
      padding: 1.25rem 1.5rem;
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.3s ease;
    }
    .result-banner.tumor {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--danger);
      box-shadow: 0 0 25px var(--danger-glow);
    }
    .result-banner.healthy {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid var(--success);
      box-shadow: 0 0 25px var(--success-glow);
    }
    .result-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }
    .result-value {
      font-size: 1.4rem;
      font-weight: 800;
      letter-spacing: -0.01em;
      margin-top: 0.2rem;
    }
    .result-banner.tumor .result-value { color: #f87171; }
    .result-banner.healthy .result-value { color: #34d399; }

    .confidence-score {
      text-align: right;
    }
    .confidence-score .pct {
      font-size: 2rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
    }
    .confidence-score span {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    /* Probability Bars */
    .bars-container {
      margin-top: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .bar-row {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .bar-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .progress-track {
      background: rgba(255, 255, 255, 0.08);
      height: 12px;
      border-radius: 9999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .progress-fill.tumor-fill {
      background: linear-gradient(90deg, #f87171, #ef4444);
    }
    .progress-fill.healthy-fill {
      background: linear-gradient(90deg, #34d399, #10b981);
    }

    /* Analytics Tab */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem;
      text-align: center;
    }
    .metric-card .num {
      font-size: 2.2rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
      margin-top: 0.4rem;
    }
    .metric-card .lbl {
      font-size: 0.8rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    .report-image-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      text-align: center;
      margin-bottom: 2rem;
    }
    .report-image-box img {
      max-width: 100%;
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Training plots grid */
    .curves-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }
    .curves-grid img {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--card-border);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="logo-badge">🧠</div>
      <div class="brand-title">
        <h1>NeuroScan AI • Brain Tumor Classifier</h1>
        <p>Convolutional Neural Network Deep Learning Diagnostics</p>
      </div>
    </div>
    <div class="badge-status">
      <div class="pulse-dot"></div>
      Model Online (TensorFlow 2.21)
    </div>
  </header>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('scanner')">🔬 Live MRI Scanner</button>
    <button class="tab-btn" onclick="switchTab('dashboard')">📊 Full Output Dashboard</button>
    <button class="tab-btn" onclick="switchTab('metrics')">📈 Model Training & Metrics</button>
  </div>

  <main>
    <!-- TAB 1: LIVE SCANNER -->
    <div id="tab-scanner" class="tab-content active">
      <div class="scanner-grid">
        <!-- Input Column -->
        <div class="card">
          <div class="card-header">
            <h2>📤 Input Brain MRI Scan</h2>
          </div>

          <div class="upload-box" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <div class="upload-icon">📁</div>
            <p><strong>Click to browse</strong> or drag & drop brain MRI scan</p>
            <p style="margin-top:0.3rem;">Supports JPG, PNG, DICOM converted (224x224 input)</p>
          </div>
          <input type="file" id="file-input" style="display:none" accept="image/*" onchange="handleFileSelect(event)">

          <div class="sample-title">Or Test With Real Kaggle Dataset MRI Scans:</div>
          <div class="sample-strip" id="sample-strip">
            <!-- Dynamic samples injected via JS -->
          </div>
        </div>

        <!-- Output Column -->
        <div class="card">
          <div class="card-header">
            <h2>🎯 Real-Time Diagnostic Output</h2>
          </div>

          <div class="preview-container" id="preview-box">
            <span class="no-mri-text">Select a sample or upload an MRI scan to view classification</span>
          </div>

          <div id="result-section" style="display:none;">
            <div class="result-banner" id="result-banner">
              <div>
                <div class="result-label">DIAGNOSTIC PREDICTION</div>
                <div class="result-value" id="result-text">TUMOR DETECTED</div>
              </div>
              <div class="confidence-score">
                <div class="pct" id="conf-text">98.9%</div>
                <span>Confidence</span>
              </div>
            </div>

            <div class="bars-container">
              <div class="bar-row">
                <div class="bar-labels">
                  <span style="color:#f87171;">Tumor Probability</span>
                  <span id="tumor-prob-lbl" style="font-family:'JetBrains Mono',monospace;">0%</span>
                </div>
                <div class="progress-track">
                  <div class="progress-fill tumor-fill" id="tumor-bar" style="width: 0%;"></div>
                </div>
              </div>

              <div class="bar-row">
                <div class="bar-labels">
                  <span style="color:#34d399;">Healthy / Non-Tumor Probability</span>
                  <span id="healthy-prob-lbl" style="font-family:'JetBrains Mono',monospace;">0%</span>
                </div>
                <div class="progress-track">
                  <div class="progress-fill healthy-fill" id="healthy-bar" style="width: 0%;"></div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- TAB 2: FULL OUTPUT DASHBOARD -->
    <div id="tab-dashboard" class="tab-content">
      <div class="report-image-box">
        <h2 style="font-size:1.3rem; margin-bottom:1rem; color:var(--accent);">High-Resolution Model Evaluation & Prediction Output</h2>
        <img src="/static/brain_tumor_detection_output.png" alt="Brain Tumor Evaluation Dashboard">
      </div>
    </div>

    <!-- TAB 3: TRAINING & METRICS -->
    <div id="tab-metrics" class="tab-content">
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="lbl">Test Set Accuracy</div>
          <div class="num">77.5%</div>
        </div>
        <div class="metric-card">
          <div class="lbl">Tumor Recall / Sensitivity</div>
          <div class="num" style="color:#34d399;">87.5%</div>
        </div>
        <div class="metric-card">
          <div class="lbl">Tumor Precision</div>
          <div class="num">77.8%</div>
        </div>
        <div class="metric-card">
          <div class="lbl">F1-Score</div>
          <div class="num">82.4%</div>
        </div>
      </div>

      <div class="curves-grid">
        <div class="card">
          <h3 style="margin-bottom:1rem; font-size:1.1rem; color:var(--accent);">Training vs Validation Accuracy</h3>
          <img src="/static/accuracy_curve.png" alt="Accuracy Curve">
        </div>
        <div class="card">
          <h3 style="margin-bottom:1rem; font-size:1.1rem; color:var(--accent);">Training vs Validation Loss</h3>
          <img src="/static/loss_curve.png" alt="Loss Curve">
        </div>
      </div>
    </div>

  </main>

  <script>
    let samples = [];

    function switchTab(tabName) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabName));
      if (targetBtn) targetBtn.classList.add('active');
      const targetContent = document.getElementById('tab-' + tabName);
      if (targetContent) targetContent.classList.add('active');
    }

    async function loadSamples() {
      const res = await fetch('/api/samples');
      samples = await res.json();
      const strip = document.getElementById('sample-strip');
      strip.innerHTML = '';
      
      samples.forEach((s, idx) => {
        const item = document.createElement('div');
        item.className = 'sample-item' + (idx === 0 ? ' selected' : '');
        item.innerHTML = `
          <img src="${s.thumbnail}" alt="${s.name}">
          <p>${s.name}</p>
        `;
        item.onclick = () => selectSample(s, item);
        strip.appendChild(item);
      });

      // Automatically trigger first sample
      if (samples.length > 0) {
        selectSample(samples[0], strip.children[0]);
      }
    }

    async function selectSample(sample, el) {
      document.querySelectorAll('.sample-item').forEach(i => i.classList.remove('selected'));
      if (el) el.classList.add('selected');

      renderMRI(sample.thumbnail);

      const res = await fetch('/api/predict_sample?id=' + encodeURIComponent(sample.id));
      const data = await res.json();
      displayResult(data);
    }

    function renderMRI(src) {
      const preview = document.getElementById('preview-box');
      preview.innerHTML = `<img src="${src}" class="mri-display" alt="Brain MRI Scan">`;
    }

    function displayResult(data) {
      document.getElementById('result-section').style.display = 'block';
      const banner = document.getElementById('result-banner');
      banner.className = 'result-banner ' + data.category;

      document.getElementById('result-text').innerText = data.result;
      document.getElementById('conf-text').innerText = data.confidence + '%';

      document.getElementById('tumor-prob-lbl').innerText = data.tumor_prob + '%';
      document.getElementById('healthy-prob-lbl').innerText = data.healthy_prob + '%';

      document.getElementById('tumor-bar').style.width = data.tumor_prob + '%';
      document.getElementById('healthy-bar').style.width = data.healthy_prob + '%';
    }

    async function handleFileSelect(evt) {
      const file = evt.target.files[0];
      if (!file) return;
      processUpload(file);
    }

    function processUpload(file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64Data = e.target.result;
        renderMRI(base64Data);

        const res = await fetch('/api/predict_upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Data })
        });
        const data = await res.json();
        displayResult(data);
      };
      reader.readAsDataURL(file);
    }

    // Drag and drop setup
    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        processUpload(e.dataTransfer.files[0]);
      }
    });

    window.onload = loadSamples;
  </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            
        elif path == "/api/samples":
            sample_list = []
            for s in SAMPLES:
                b64 = image_to_base64(s["path"])
                sample_list.append({
                    "id": s["id"],
                    "name": s["name"],
                    "thumbnail": b64,
                    "expected": s["expected"]
                })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sample_list).encode("utf-8"))

        elif path == "/api/predict_sample":
            query = urllib.parse.parse_qs(url.query)
            sample_id = query.get("id", [""])[0]
            target = next((s for s in SAMPLES if s["id"] == sample_id), None)
            if target and os.path.exists(target["path"]):
                img = Image.open(target["path"])
                pred = predict_pil_image(img)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(pred).encode("utf-8"))
            else:
                self.send_error(404, "Sample not found")

        elif path.startswith("/static/"):
            filename = os.path.basename(path)
            file_path = filename
            if filename == "accuracy_curve.png":
                file_path = "models/accuracy_curve.png"
            elif filename == "loss_curve.png":
                file_path = "models/loss_curve.png"
            elif filename == "brain_tumor_detection_output.png":
                file_path = "brain_tumor_detection_output.png"
                
            if os.path.exists(file_path):
                self.send_response(200)
                if filename.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"File {filename} not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/predict_upload":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                img_data_url = data.get("image", "")
                header, base64_str = img_data_url.split(",", 1)
                img_bytes = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(img_bytes))
                pred = predict_pil_image(img)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(pred).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

def run():
    server = HTTPServer(("127.0.0.1", PORT), RequestHandler)
    print(f"Server started at http://localhost:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    run()
