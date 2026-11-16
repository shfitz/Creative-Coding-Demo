let capture; // object to hold the video grabber 
let phase = 1;
let creature;
let minDepth, maxDepth; // depth thresholds (0-255, brighter = closer)

let depthEstimator;

let depthMap;

// offscreen buffers: one for the mask, one for the camera frame we cut
let maskBuffer;
let sceneBuffer;

// slider UI for the depth band + feather
let minSlider, maxSlider, featherSlider;

async function setup() {
  createCanvas(800, 600);

  // start video grabber
  depthEstimator = await ml5.depthEstimation();

  capture = await createCapture(VIDEO);
  capture.size(800, 600);
  capture.hide(); // hide HTML video
  depthEstimator.estimateStart(capture, gotResults);

  // buffers are the same size as the canvas
  maskBuffer = createGraphics(800, 600);
  sceneBuffer = createGraphics(800, 600);

  // record work
  minSlider = createSlider(0, 255, 200);
  maxSlider = createSlider(0, 255, 255);
  featherSlider = createSlider(0, 15, 3); // 0 = hard edge
  minSlider.position(10, height + 10);
  maxSlider.position(10, height + 30);
  featherSlider.position(10, height + 50);
  createP("min (near)");
  createP("max (far)");
  createP("feather");
}

function draw() {
  switch(phase){
    case 0:
      // grab the background
      break;
    case 1:
      // track object 
      trackTheDepth();
      break;
    case 3:
      // attract screen
  }
}

function trackTheDepth(){
  if (!depthMap) {
    return; // model not ready yet
  }

  // Read the sliders into the depth band.
  minDepth = minSlider.value();
  maxDepth = maxSlider.value();

  // 1. Build the mask from the DEPTH MAP's brightness. ml5's depth map is
  //    grayscale where brighter = closer, so we keep only the pixels inside
  //    [minDepth, maxDepth] — i.e. objects at the distance we care about.
  maskBuffer.clear();
  maskFromDepth(depthMap.image, minDepth, maxDepth, maskBuffer);
  // Feather the mask edge: blur softens the hard 255|0 boundary into a
  // gradient, so the camera fades in from the background instead of a
  // razor cut. Radius 0 = no blur (hard edge).
  const feather = featherSlider.value();
  if (feather > 0) {
    maskBuffer.filter(BLUR, feather);
  }
  // 2. Draw the LIVE CAMERA into the scene buffer.
  sceneBuffer.clear();
  sceneBuffer.image(capture, 0, 0, 800, 600);

  // 3. Cut the near objects' pixels out of the camera frame.
  cutWithMask(sceneBuffer, maskBuffer);

  // 4. Show the result — background is transparent, black shows through.
  background(0);
  image(sceneBuffer, 0, 0, 800, 600);
}

// Threshold a depth-map image into a white / transparent mask
function maskFromDepth(depthImage, lo, hi, out) {
  out.image(depthImage, 0, 0, out.width, out.height);
  out.loadPixels();
  const dst = out.pixels;

  for (let i = 0; i < dst.length; i += 4) {
    // brightness of the buffer pixel = depth value (grayscale, r = depth)
    const d = dst[i];
    const keep = d >= lo && d <= hi;

    dst[i] = 255;       // white mask
    dst[i + 1] = 255;
    dst[i + 2] = 255;
    dst[i + 3] = keep ? 255 : 0; // opaque = keep, transparent = remove
  }

  out.updatePixels();
}

// Cut an offscreen buffer's drawn content with an ALPHA mask using raw 2D
// compositing. "destination-in" keeps existing pixels only where the source
// (the mask) has alpha. Opaque white = kept, transparent = erased.
function cutWithMask(scene, mask) {
  const ctx = scene.drawingContext;

  ctx.save();
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(mask.canvas, 0, 0, 800, 600);
  ctx.restore();
}

function gotResults(result) {
  // Store the latest result in the global variable depthMap
  depthMap = result;
}
