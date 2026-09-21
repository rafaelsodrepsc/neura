// --- dataset: duas espirais entrelaçadas -----------------------------------

function generateSpiral(count, rotationOffset, label) {
  const points = [];
  for (let i = 0; i < count; i++) {
    const r = (i / count) * 5;
    const t = 1.75 * (i / count) * 2 * Math.PI + rotationOffset;
    const noise = (Math.random() - 0.5) * 0.9;
    const x = (r * Math.sin(t) + noise) / 6;
    const y = (r * Math.cos(t) + noise) / 6;
    points.push({ x, y, label });
  }
  return points;
}

function buildDataset() {
  return [...generateSpiral(110, 0, 0), ...generateSpiral(110, Math.PI, 1)];
}

// --- rede neural: MLP com backprop manual -----------------------------------

const tanh = Math.tanh;
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

const MOMENTUM = 0.9;

class NeuralNetwork {
  constructor(sizes) {
    this.sizes = sizes;
    this.W = [];
    this.b = [];
    this.vW = [];
    this.vB = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l], fanOut = sizes[l + 1];
      const scale = Math.sqrt(2 / fanIn);
      const w = [];
      for (let j = 0; j < fanOut; j++) {
        const row = [];
        for (let k = 0; k < fanIn; k++) row.push((Math.random() * 2 - 1) * scale);
        w.push(row);
      }
      this.W.push(w);
      this.b.push(new Array(fanOut).fill(0));
      this.vW.push(w.map((row) => row.map(() => 0)));
      this.vB.push(new Array(fanOut).fill(0));
    }
  }

  forward(x) {
    const activations = [x];
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const W = this.W[l], b = this.b[l];
      const isOutput = l === this.W.length - 1;
      const z = W.map((row, j) => row.reduce((s, w, k) => s + w * a[k], 0) + b[j]);
      a = isOutput ? z.map(sigmoid) : z.map(tanh);
      activations.push(a);
    }
    return activations;
  }

  predict(x) {
    const activations = this.forward(x);
    return activations[activations.length - 1][0];
  }

  trainStep(batch, lr) {
    const gradW = this.W.map((w) => w.map((row) => row.map(() => 0)));
    const gradB = this.b.map((b) => b.map(() => 0));
    let totalLoss = 0;
    const L = this.W.length - 1;

    for (const { x: px, y: py, label } of batch) {
      const activations = this.forward([px, py]);
      const output = activations[activations.length - 1][0];
      const y = label;
      const eps = 1e-9;
      totalLoss += -(y * Math.log(output + eps) + (1 - y) * Math.log(1 - output + eps));

      let delta = [output - y]; // sigmoid + cross-entropy: dL/dz = output - y

      for (let j = 0; j < gradW[L].length; j++) {
        for (let k = 0; k < gradW[L][j].length; k++) {
          gradW[L][j][k] += delta[j] * activations[L][k];
        }
        gradB[L][j] += delta[j];
      }

      let deltaNext = delta;
      for (let l = L - 1; l >= 0; l--) {
        const a = activations[l + 1];
        const Wnext = this.W[l + 1];
        const newDelta = new Array(this.sizes[l + 1]).fill(0);
        for (let k = 0; k < Wnext.length; k++) {
          for (let j = 0; j < Wnext[k].length; j++) {
            newDelta[j] += Wnext[k][j] * deltaNext[k];
          }
        }
        for (let j = 0; j < newDelta.length; j++) newDelta[j] *= (1 - a[j] * a[j]);

        for (let j = 0; j < gradW[l].length; j++) {
          for (let k = 0; k < gradW[l][j].length; k++) {
            gradW[l][j][k] += newDelta[j] * activations[l][k];
          }
          gradB[l][j] += newDelta[j];
        }
        deltaNext = newDelta;
      }
    }

    const n = batch.length;
    for (let l = 0; l < this.W.length; l++) {
      for (let j = 0; j < this.W[l].length; j++) {
        for (let k = 0; k < this.W[l][j].length; k++) {
          this.vW[l][j][k] = MOMENTUM * this.vW[l][j][k] - lr * gradW[l][j][k] / n;
          this.W[l][j][k] += this.vW[l][j][k];
        }
        this.vB[l][j] = MOMENTUM * this.vB[l][j] - lr * gradB[l][j] / n;
        this.b[l][j] += this.vB[l][j];
      }
    }
    return totalLoss / n;
  }
}

// --- estado --------------------------------------------------------------

const HIDDEN_SIZES = [2, 12, 12, 1];
let data = buildDataset();
let net = new NeuralNetwork(HIDDEN_SIZES);
let running = true;
let epoch = 0;
let lastLoss = null;
let lossHistory = [];

// --- canvas refs -----------------------------------------------------------

const fieldCanvas = document.getElementById("fieldCanvas");
const fieldCtx = fieldCanvas.getContext("2d");
const netCanvas = document.getElementById("netCanvas");
const netCtx = netCanvas.getContext("2d");
const lossCanvas = document.getElementById("lossCanvas");
const lossCtx = lossCanvas.getContext("2d");

const statEpoch = document.getElementById("statEpoch");
const statLoss = document.getElementById("statLoss");
const statAcc = document.getElementById("statAcc");
const playBtn = document.getElementById("playBtn");
const resetBtn = document.getElementById("resetBtn");
const speedRange = document.getElementById("speedRange");
const lrRange = document.getElementById("lrRange");

function readTokens() {
  const style = getComputedStyle(document.documentElement);
  return {
    text: style.getPropertyValue("--text").trim(),
    textDim: style.getPropertyValue("--text-dim").trim(),
    accentA: style.getPropertyValue("--accent-a").trim(),
    accentB: style.getPropertyValue("--accent-b").trim(),
    gridLine: style.getPropertyValue("--grid-line").trim(),
    surfaceBorder: style.getPropertyValue("--surface-border").trim(),
  };
}

// --- desenho: campo de decisão ---------------------------------------------

const FIELD_RES = 48;
const FIELD_RANGE = 1.1;

function hexToRgb(hex) {
  const m = hex.replace("#", "").match(/.{1,2}/g);
  return m.map((h) => parseInt(h, 16));
}

function drawField() {
  const tokens = readTokens();
  const w = fieldCanvas.width, h = fieldCanvas.height;
  fieldCtx.clearRect(0, 0, w, h);

  const rgbA = hexToRgb(tokens.accentA.startsWith("#") ? tokens.accentA : "#6d3bff");
  const rgbB = hexToRgb(tokens.accentB.startsWith("#") ? tokens.accentB : "#00a887");

  const cell = w / FIELD_RES;
  for (let gx = 0; gx < FIELD_RES; gx++) {
    for (let gy = 0; gy < FIELD_RES; gy++) {
      const x = (gx / (FIELD_RES - 1)) * (2 * FIELD_RANGE) - FIELD_RANGE;
      const y = (gy / (FIELD_RES - 1)) * (2 * FIELD_RANGE) - FIELD_RANGE;
      const p = net.predict([x, y]);
      const t = Math.abs(p - 0.5) * 2; // confiança
      const rgb = p > 0.5 ? rgbB : rgbA;
      const alpha = 0.10 + t * 0.28;
      fieldCtx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
      fieldCtx.fillRect(gx * cell, gy * cell, cell + 1, cell + 1);
    }
  }

  for (const { x, y, label } of data) {
    const px = ((x + FIELD_RANGE) / (2 * FIELD_RANGE)) * w;
    const py = ((y + FIELD_RANGE) / (2 * FIELD_RANGE)) * h;
    fieldCtx.beginPath();
    fieldCtx.arc(px, py, 3.4, 0, Math.PI * 2);
    fieldCtx.fillStyle = label === 0 ? tokens.accentA : tokens.accentB;
    fieldCtx.fill();
  }
}

// --- desenho: diagrama da rede ----------------------------------------------

function drawNetwork() {
  const tokens = readTokens();
  const w = netCanvas.width, h = netCanvas.height;
  netCtx.clearRect(0, 0, w, h);

  const sizes = HIDDEN_SIZES;
  const margin = 40;
  const colX = sizes.map((_, i) => margin + (i * (w - 2 * margin)) / (sizes.length - 1));
  const positions = sizes.map((count, i) => {
    const gap = h / (count + 1);
    return new Array(count).fill(0).map((_, j) => ({ x: colX[i], y: gap * (j + 1) }));
  });

  // arestas
  for (let l = 0; l < net.W.length; l++) {
    const W = net.W[l];
    for (let j = 0; j < W.length; j++) {
      for (let k = 0; k < W[j].length; k++) {
        const weight = W[j][k];
        const from = positions[l][k];
        const to = positions[l + 1][j];
        const mag = Math.min(Math.abs(weight), 2.2) / 2.2;
        netCtx.globalAlpha = 0.08 + mag * 0.55;
        netCtx.lineWidth = 0.6 + mag * 2;
        netCtx.strokeStyle = weight >= 0 ? tokens.accentA : tokens.accentB;
        netCtx.beginPath();
        netCtx.moveTo(from.x, from.y);
        netCtx.lineTo(to.x, to.y);
        netCtx.stroke();
      }
    }
  }
  netCtx.globalAlpha = 1;

  // nós
  for (const layer of positions) {
    for (const { x, y } of layer) {
      netCtx.beginPath();
      netCtx.arc(x, y, 5, 0, Math.PI * 2);
      netCtx.fillStyle = tokens.text;
      netCtx.fill();
    }
  }
}

// --- desenho: curva de loss --------------------------------------------------

function drawLoss() {
  const tokens = readTokens();
  const w = lossCanvas.width, h = lossCanvas.height;
  lossCtx.clearRect(0, 0, w, h);

  if (lossHistory.length < 2) return;

  const maxLoss = Math.max(...lossHistory, 0.05);
  const step = w / (lossHistory.length - 1);

  lossCtx.beginPath();
  lossHistory.forEach((loss, i) => {
    const x = i * step;
    const y = h - (loss / maxLoss) * (h - 10) - 4;
    if (i === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  });
  lossCtx.strokeStyle = tokens.accentA;
  lossCtx.lineWidth = 2;
  lossCtx.stroke();

  lossCtx.strokeStyle = tokens.gridLine;
  lossCtx.lineWidth = 1;
  lossCtx.beginPath();
  lossCtx.moveTo(0, h - 4);
  lossCtx.lineTo(w, h - 4);
  lossCtx.stroke();
}

// --- loop de treino + render --------------------------------------------------

function accuracy() {
  let correct = 0;
  for (const { x, y, label } of data) {
    const p = net.predict([x, y]);
    if ((p > 0.5 ? 1 : 0) === label) correct++;
  }
  return correct / data.length;
}

function tick() {
  if (running) {
    const stepsPerFrame = Number(speedRange.value);
    const lr = Number(lrRange.value) * 0.005; // momentum multiplica o passo efetivo em ~10x
    for (let s = 0; s < stepsPerFrame; s++) {
      lastLoss = net.trainStep(data, lr);
      epoch++;
    }
    lossHistory.push(lastLoss);
    if (lossHistory.length > 300) lossHistory.shift();

    statEpoch.textContent = epoch;
    statLoss.textContent = lastLoss.toFixed(4);
    if (epoch % 5 === 0 || epoch < 5) {
      statAcc.textContent = `${(accuracy() * 100).toFixed(0)}%`;
    }
  }

  drawField();
  drawNetwork();
  drawLoss();
  requestAnimationFrame(tick);
}

// --- controles -----------------------------------------------------------

playBtn.addEventListener("click", () => {
  running = !running;
  playBtn.textContent = running ? "pausar" : "continuar";
});

resetBtn.addEventListener("click", () => {
  data = buildDataset();
  net = new NeuralNetwork(HIDDEN_SIZES);
  epoch = 0;
  lastLoss = null;
  lossHistory = [];
  statEpoch.textContent = "0";
  statLoss.textContent = "–";
  statAcc.textContent = "–";
  running = true;
  playBtn.textContent = "pausar";
});

requestAnimationFrame(tick);
