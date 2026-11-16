const SETUP = 1;
const TRACK = 2;

const DEPTH_MODEL = "onnx-community/depth-anything-v2-small";

let state = SETUP;
let capture = null;

// depth
let depthPipe = null;
let depthDevice = "";
let depthWanted = false; // the loop should be running
let depthLoopActive = false; // the loop is running, possibly mid-inference
let depthStoppedAt = -9999;
let depthStretch = null; // smoothed min/max, so a fixed band keeps its meaning
let depthResult = null;
let bandSeeded = false;

// buffers
let maskBuffer; // depth band -> white/transparent alpha mask
let cutBuffer; // the frame cut with that mask
let frameBuffer; // one snapshot of the live frame: drawn, scanned and cut from
let inferCanvas; // the snapshot handed to the model
let inferCtx;

// sprite
let sprite = null;
let spriteW = 0;
let spriteH = 0;

// tracking
let peakX = 0; // brightest pixel in the current frame
let peakY = 0;
let peakBrightness = 0;
let spriteX = 0; // where the sprite is drawn
let spriteY = 0;
let tracking = false;

// equally bright neighbours don't show. At 30 fps a window of 5 lags ~2 frames.
const TRACK_WINDOW = 10;
let peakHistoryX = [];
let peakHistoryY = [];

// ui
let setupPanel;
let trackPanel;
let bandLoSlider;
let bandHiSlider;
let featherSlider;
let scaleSlider;
let problem = "";

async function setup() {
  createCanvas(800, 600);
  pixelDensity(1); 

  maskBuffer = createGraphics(width, height);
  maskBuffer.pixelDensity(1);
  cutBuffer = createGraphics(width, height);
  cutBuffer.pixelDensity(1);
  frameBuffer = createGraphics(width, height);
  frameBuffer.pixelDensity(1);

  inferCanvas = document.createElement("canvas");
  inferCanvas.width = width;
  inferCanvas.height = height;
  inferCtx = inferCanvas.getContext("2d", { willReadFrequently: true });

  buildUI();
  showPanels();

  try {
    await loadDepthModel();
    capture = await openCamera();
    startDepth();
  } catch (err) {
    problem = "could not start the camera or the depth model — " + err.message;
  }
}

function draw() {
  background(0);

  if (problem) {
    textAlign(CENTER, CENTER);
    textSize(14);
    noStroke();
    fill(255, 120, 120);
    text(problem, width / 2, height / 2, width - 80);
    return;
  }

  switch (state) {
    case SETUP:
      drawSetup();
      break;
    case TRACK:
      drawTrack();
      break;
  }

  drawHUD();
}

/* depth */

async function openCamera() {
  const cam = await createCapture(VIDEO);
  cam.size(width, height);
  cam.hide();
  return cam;
}

async function loadDepthModel() {
  const HF = await hfModule();

  depthDevice = "wasm";
  if (navigator.gpu) depthDevice = "webgpu";

  // wasm without these options; webgpu is the faster path
  let options = {};
  if (depthDevice === "webgpu") options = { device: "webgpu", dtype: "fp32" };

  depthPipe = await HF.pipeline("depth-estimation", DEPTH_MODEL, options);
}

async function hfModule() {
  for (let i = 0; i < 600; i++) {
    if (window.HF) return window.HF;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("transformers.js never loaded......");
}

// separate so a test can stub the model
async function runDepthModel(input) {
  const output = await depthPipe(input);
  return output.predicted_depth; // { data, dims }
}

function startDepth() {
  if (depthWanted || !depthPipe || !capture) return;
  depthWanted = true;
  if (!depthLoopActive) depthLoop();
}

function stopDepth() {
  if (!depthWanted) return;
  depthWanted = false;
  depthStoppedAt = millis();
}

async function depthLoop() {
  depthLoopActive = true;
  while (depthWanted) {
    try {
      await estimateOnce();
    } catch (err) {
      problem = "depth estimation failed — " + err.message;
      depthWanted = false;
    }
    if (depthWanted) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  depthLoopActive = false;
  depthStoppedAt = millis();
}

async function estimateOnce() {
  if (!capture) return;

  try {
    drawMirrored(inferCtx, captureElement());
  } catch (err) {
    return; // camera hasn't delivered a frame yet
  }

  const tensor = await runDepthModel(inferCanvas);
  if (!tensor || !tensor.data) return;

  const image = depthImageFromTensor(tensor);
  if (!image) return;

  frameBuffer.clear();
  frameBuffer.drawingContext.drawImage(inferCanvas, 0, 0, width, height);
  depthResult = { image, sourceFrame: frameBuffer };

  if (!bandSeeded) seedBand(image);
}

// DepthAnything's values are larger when something is closer to the camera
function depthImageFromTensor(tensor) {
  const dims = tensor.dims;
  const raw = tensor.data;

  let h = dims[0];
  let w = dims[1];
  if (dims.length === 3) {
    h = dims[1];
    w = dims[2];
  }

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] < min) min = raw[i];
    if (raw[i] > max) max = raw[i];
  }

  if (!depthStretch) {
    depthStretch = { min, max };
  } else {
    depthStretch.min += (min - depthStretch.min) * 0.25;
    depthStretch.max += (max - depthStretch.max) * 0.25;
  }

  const lo = depthStretch.min;
  let range = depthStretch.max - lo;
  if (range === 0) range = 1;

  const image = createImage(w, h);
  image.pixelDensity(1);
  image.loadPixels();
  const px = image.pixels;

  let brightest = 0;
  for (let i = 0; i < w * h; i++) {
    let v = Math.round(((raw[i] - lo) / range) * 255);
    if (v < 0) v = 0;
    else if (v > 255) v = 255;

    px[i * 4] = v;
    px[i * 4 + 1] = v;
    px[i * 4 + 2] = v;
    px[i * 4 + 3] = 255;

    if (v > brightest) brightest = v;
  }
  image.updatePixels();

  if (brightest === 0) return null;
  return image;
}

/* state 1 */

function drawSetup() {

  if (!depthWanted && capture && depthPipe && millis() - depthStoppedAt > 300) {
    startDepth();
  }

  if (!depthResult) {
    textAlign(CENTER, CENTER);
    textSize(14);
    noStroke();
    fill(180);
    text(waitingText(), width / 2, height / 2, width - 80);
    return;
  }

  const frame = sourceFrame();
  if (!frame) return;

  paintMask();
  cutBuffer.clear();
  cutBuffer.image(frame, 0, 0, width, height);
  cutWithMask(cutBuffer, maskBuffer);
  image(cutBuffer, 0, 0, width, height);

  // depth map preview
  const mw = 180;
  const mh = 135;
  const mx = width - mw - 12;
  const my = 12;
  image(depthResult.image, mx, my, mw, mh);
  noFill();
  stroke(255, 90);
  rect(mx, my, mw, mh);
  noStroke();
  fill(220);
  textAlign(LEFT, TOP);
  textSize(11);
  text("depth map | brighter = nearer", mx, my + mh + 5);

  if (depthDevice === "wasm") {
    textAlign(CENTER, CENTER);
    textSize(13);
    fill(255, 220, 140);
    text("no WebGPU in this browser — depth is taking about 10 s a frame", width / 2, height - 60);
  }
}


function paintMask() {
  maskBuffer.clear();
  maskFromDepth(depthResult.image, bandLoSlider.value(), bandHiSlider.value(), maskBuffer);

  const feather = featherSlider.value();
  if (feather > 0) maskBuffer.filter(BLUR, feather);
}

/* state 2 */

function drawTrack() {
  if (capture) {
    frameBuffer.clear();
    drawMirrored(frameBuffer.drawingContext, captureElement());
    image(frameBuffer, 0, 0, width, height);
  }

  tracking = false;
  if (capture) {
    const peak = findBrightestPixel();
    if (peak) {
      peakX = peak.x;
      peakY = peak.y;
      peakBrightness = peak.brightness;
      tracking = true;
      rememberPeak(peak.x, peak.y);
    }
  }

  const smoothed = smoothedPeak();
  if (smoothed) {
    spriteX = smoothed.x;
    spriteY = smoothed.y;
  }

  if (sprite) {
    const s = scaleSlider.value();
    const dw = spriteW * s;
    const dh = spriteH * s;
    image(sprite, spriteX - dw / 2, spriteY - dh / 2, dw, dh);
  }

}

function rememberPeak(x, y) {
  peakHistoryX.push(x);
  peakHistoryY.push(y);
  if (peakHistoryX.length > TRACK_WINDOW) peakHistoryX.shift();
  if (peakHistoryY.length > TRACK_WINDOW) peakHistoryY.shift();
}

function forgetPeaks() {
  peakHistoryX = [];
  peakHistoryY = [];
}

function smoothedPeak() {
  if (peakHistoryX.length === 0) return null;

  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < peakHistoryX.length; i++) {
    sumX += peakHistoryX[i];
    sumY += peakHistoryY[i];
  }

  return { x: sumX / peakHistoryX.length, y: sumY / peakHistoryY.length };
}

function findBrightestPixel() {
  const d = frameBuffer.drawingContext.getImageData(0, 0, width, height).data;

  let best = -1;
  let bestX = 0;
  let bestY = 0;

  for (let i = 0; i < d.length; i += 4) {
    const brightness = luminance(d[i], d[i + 1], d[i + 2]);
    if (brightness > best) {
      best = brightness;
      const index = i / 4;
      bestX = index % width;
      bestY = Math.floor(index / width);
    }
  }

  if (best < 0) return null;
  return { x: bestX + 0.5, y: bestY + 0.5, brightness: best };
}

/* capture */

function captureSprite() {
  if (state !== SETUP) return;
  if (!depthResult) return;

  const frame = sourceFrame();
  if (!frame) return;

  // a band seeded from one frame can select nothing once the scene changes
  if (!bandHasDepth()) seedBand(depthResult.image);

  paintMask();
  cutBuffer.clear();
  cutBuffer.image(frame, 0, 0, width, height);
  cutWithMask(cutBuffer, maskBuffer);

  const box = alphaBounds(cutBuffer);
  if (!box) return;

  if (sprite) sprite.remove();
  sprite = createGraphics(box.w, box.h);
  sprite.pixelDensity(1);
  sprite.drawingContext.drawImage(
    cutBuffer.canvas,
    box.x,
    box.y,
    box.w,
    box.h,
    0,
    0,
    box.w,
    box.h
  );
  spriteW = box.w;
  spriteH = box.h;

  stopDepth(); // nothing in TRACK needs depth
  state = TRACK;
  forgetPeaks();

  spriteX = width / 2;
  spriteY = height / 2;
  peakX = width / 2;
  peakY = height / 2;

  showPanels();
}

function backToSetup() {
  state = SETUP;
  if (sprite) {
    sprite.remove();
    sprite = null;
  }
  spriteW = 0;
  spriteH = 0;
  depthResult = null;
  forgetPeaks();
  showPanels();
}

function seedBand(depthImage) {
  if (!depthImage) return;

  depthImage.loadPixels();
  const px = depthImage.pixels;

  let peak = 0;
  for (let i = 0; i < px.length; i += 4) {
    if (px[i] > peak) peak = px[i];
  }
  if (peak <= 0) return;

  setSlider(bandLoSlider, max(0, peak - 60));
  setSlider(bandHiSlider, 255);
  bandSeeded = true;
}

// is any of the depth map inside the sliders' band?
function bandHasDepth() {
  const lo = bandLoSlider.value();
  const hi = bandHiSlider.value();

  depthResult.image.loadPixels();
  const px = depthResult.image.pixels;

  for (let i = 0; i < px.length; i += 4) {
    if (px[i] >= lo && px[i] <= hi) return true;
  }
  return false;
}

/* pixel work */

function maskFromDepth(depthImage, lo, hi, out) {
  out.image(depthImage, 0, 0, out.width, out.height);
  out.loadPixels();
  const px = out.pixels;

  for (let i = 0; i < px.length; i += 4) {
    const depth = px[i]; // grayscale, so r = g = b

    // opaque = keep, transparent = drop
    let alpha = 0;
    if (depth >= lo && depth <= hi) alpha = 255;

    px[i] = 255;
    px[i + 1] = 255;
    px[i + 2] = 255;
    px[i + 3] = alpha;
  }

  out.updatePixels();
}

function cutWithMask(target, mask) {
  const ctx = target.drawingContext;
  ctx.save();
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(mask.canvas, 0, 0, target.width, target.height);
  ctx.restore();
}

function alphaBounds(buffer) {
  buffer.loadPixels();
  const px = buffer.pixels;

  let minX = buffer.width;
  let minY = buffer.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < buffer.height; y++) {
    for (let x = 0; x < buffer.width; x++) {
      if (px[(y * buffer.width + x) * 4 + 3] > 8) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function sourceFrame() {
  if (depthResult && depthResult.sourceFrame) return depthResult.sourceFrame;
  return capture;
}

function captureElement() {
  return capture.elt;
}

function drawMirrored(ctx, source) {
  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(source, 0, 0, width, height);
  ctx.restore();
}

function waitingText() {
  if (!capture) return "starting the camera…";
  if (!depthPipe) return "loading the depth model…";

  const el = capture.elt;
  if (el && el.tagName === "VIDEO" && el.readyState < 2) {
    return "allow camera access for this page";
  }
  return "working out the depth map…";
}

function luminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/* ui */

function buildUI() {
  setupPanel = createDiv().addClass("panel");

  createSpan("Position your creation in the camera. Use the sliders so it is the only thing visible against the black background").addClass("title").parent(setupPanel);
  bandLoSlider = sliderRow(setupPanel, "nearest depth", 0, 255, 200, 1);
  bandHiSlider = sliderRow(setupPanel, "furthest depth", 0, 255, 255, 1);
  featherSlider = sliderRow(setupPanel, "feather", 0, 15, 3, 0.5);

  const captureBtn = createButton("capture image (space)").parent(setupPanel);
  captureBtn.mousePressed(captureSprite);

  trackPanel = createDiv().addClass("panel");

  createSpan("Follow the brightest pixels").addClass("title").parent(trackPanel);
  scaleSlider = sliderRow(trackPanel, "Scale your image", 0.05, 3, 1, 0.05);

  const backBtn = createButton("Press C to capture a new image").parent(trackPanel);
  backBtn.mousePressed(backToSetup);
}

// label + slider + readout on one line
function sliderRow(panel, label, lo, hi, value, step) {
  const row = createDiv().addClass("row").parent(panel);
  createSpan(label).addClass("label").parent(row);
  const slider = createSlider(lo, hi, value, step).parent(row);

  let decimals = 0;
  if (step < 1) decimals = 2;
  slider.decimals = decimals;

  const readout = createSpan(format(slider.value(), decimals)).addClass("value").parent(row);
  slider.readout = readout;
  slider.input(() => readout.html(format(slider.value(), decimals)));

  return slider;
}

// set a slider and its readout together — p5 only fires input() on a real drag
function setSlider(slider, value) {
  slider.value(value);
  slider.readout.html(format(value, slider.decimals));
}

function format(value, decimals) {
  return Number(value).toFixed(decimals);
}

function showPanels() {
  let setupDisplay = "none";
  let trackDisplay = "none";
  if (state === SETUP) setupDisplay = "";
  if (state === TRACK) trackDisplay = "";

  setupPanel.elt.style.display = setupDisplay;
  trackPanel.elt.style.display = trackDisplay;
}

function drawHUD() {
  noStroke();
  textAlign(LEFT, BOTTOM);
  textSize(12);
  fill(0, 150);
  rect(0, height - 26, width, 26);
  fill(235);

  if (state === SETUP) {
    text("SETUP  Spacebar to capture", 12, height - 8);
  } else {
    text(
      "TRACKING at " +
        round(peakX) +
        ", " +
        round(peakY) +
        " · scale " +
        format(scaleSlider.value(), scaleSlider.decimals) +
        " · Press 'c' to take a new photo",
      12,
      height - 8
    );
  }
}

/* input */

function keyPressed() {
  if (key === " ") {
    if (state === SETUP) captureSprite();
    return false; // keep space from scrolling the page
  }
  if (key === "c" || key === "C") {
    if (state === TRACK) backToSetup();
  }
  if (key === "s" || key === "S") {
    saveCanvas("brightsprite-" + Date.now(), "png");
  }
}
