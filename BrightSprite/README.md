# BrightSprite

Cut a piece out of an image using a depth band, keep it as a sprite, then make
that sprite ride the brightest pixel in the live camera feed.

Two states:

**1 · position, then capture.** Live camera, plus a depth map of the same frame,
estimated by DepthAnything (transformers.js). Three sliders pull a brightness
band out of that map, threshold it into an alpha mask, and feather the edge. The
mask cuts those pixels out of the camera frame. Press SPACE (or the button) and
the cut-out becomes a sprite — trimmed to the pixels it actually kept, so the
spare transparency doesn't come along with it.

**2 · follow the brightest pixel.** A snapshot of the camera feed fills the
canvas. Every frame the sketch scans that snapshot for the brightest pixel, puts
the sprite on it, and scales it with the scale slider.

Keys: `space` capture · `c` back to capture · `s` save a PNG of the frame.

## Running it

Camera access and the model both want a real origin, so serve the folder rather
than opening the file:

```bash
python3 -m http.server 8000     # then http://localhost:8000/
# or: npx http-server
# or: VS Code Live Server
```

Loads p5.js 2.3.4 and transformers.js from CDNs, so the first run wants a
network connection and pulls the depth model (~25 MB, cached after that).

**Use a browser with WebGPU** — Chromium/Vivaldi on this Mac have it. Measured at
800×600 on an Apple M6: 0.3–0.6 s per depth map on WebGPU, about 11 s per map on
the WASM fallback (which also blocks the page while it works, so state 1 becomes
a slideshow). The sketch says so on screen when it lands there. State 2 never
touches the depth model, so tracking is full speed either way.

## The depth map

**The depth map is `onnx-community/depth-anything-v2-small`**, a general-purpose
monocular model, run directly through transformers.js. It is not ml5 — see below.

**Brighter is nearer.** DepthAnything's raw values are larger where a surface is
nearer, so a plain min/max stretch lands on the convention the sliders are
labelled with. Checked against a photograph with a known near end: bottom third
(the ledge, the person) averaged 136, top third (skyline, sky) averaged 2. The
stretch is smoothed across frames, otherwise the whole map rescales whenever the
scene changes and a fixed band quietly stops selecting the same depth slice.

**The band seeds itself on the first depth map** — low edge 60 levels below the
brightest value found, high edge 255 — so the sketch opens with something in the
frame rather than an empty cut. Drag from there. The seed comes from one frame
while the depth stretch is smoothed, so if a change of scene leaves the band
selecting nothing, a capture re-seeds it from the map in hand and retries once.

**The video is mirrored**, in both states. The flip happens where the camera is
copied into a canvas, not on the display — the depth map, the cut and the tracked
peak are all made from that same flipped copy, so they share the coordinates you
see. Flipping the canvas instead would mirror the sprite, the depth monitor and
the HUD text along with the feed, and leave tracking in sensor coordinates.

**The cut comes from the frame the map was estimated from**, not the live video.
The model runs at a few frames a second and the camera at thirty, so a mask built
from the current map and pasted over the current video frame tears at the edges.
Each estimate snapshots the frame first and cuts from that same snapshot.

**Depth stops in state 2.** Nothing needs it once the sprite exists, and running
it costs a third of the GPU. The loop is restarted from `drawSetup` once it has
wound down — checked per frame rather than with a timer, so a throttled tab
cannot leave it stopped.

### Why not ml5

ml5's DepthEstimation module is generic depth estimation by design, and its
[reference](https://docs.ml5js.org/#/reference/depthestimation) says so: *"By
default, ml5.js DepthEstimation uses the Depth Anything model."* The catch is
that the docs describe the repo's main branch, and no release has it yet:

- The DepthAnything support merged to main on **2026-11-09**; the published build
  is **1.4.0, released 2026-08-07**. `https://unpkg.com/ml5@1/dist/ml5.js` — the
  URL the docs' own Quick Start gives you — resolves to 1.4.0, whose depth
  estimator is built as `handleModelName(modelName, ["ARPortraitDepth"],
  "ARPortraitDepth", …)`. One accepted model, and it is the default.
- Asking for `'DepthAnything'` there does not fail loudly: the validator warns and
  falls back, so you get ARPortraitDepth while believing otherwise.
- ARPortraitDepth is portrait-domain — ml5's own docs call it *"unreliable for any
  other kind of image"*. Measured across four photographs its off-portrait maps
  collapse to a **single value** (a phone macro and a skyline photo both returned
  1 distinct value across the whole map), and with the default person mask on,
  three non-portrait photos came back 99–100% zeroed. A live camera on this Mac
  gave a map that was 86% zeroed.

A band needs depth across the frame, so this sketch runs the model ml5 documents
and main implements, directly, until a release ships it. Switching back later is
two functions: `loadDepthModel()` and `runDepthModel()` are the only places that
touch the model.

## The tracker

It reports the brightest pixel in the frame, and nothing else. Three things were
learned the hard way, all measured on a live feed:

- **A luminance centroid over a tolerance window is not the brightest pixel.** A
  real frame has thousands of pixels within a few levels of the peak — 8,696 of
  480,000 at tolerance 20 on one test frame — so the centroid sits in the middle
  of the bright area: 52 px from the highlight. The sketch takes the peak pixel
  itself.
- **Scan the snapshot, not a downscaled copy.** A cheap 320×240 first pass with a
  local full-resolution refine still missed the peak by 12 px, because averaging
  in the downscale moves the argmax whenever a small sharp highlight sits in a
  broadly bright scene. One pass over the full frame is exact, and costs ~4 ms.
- **Take the snapshot from the video element, not through p5.** p5 draws media
  through its own frame-gated pipeline, and its copy of a `<video>` can be a
  frame or more behind the element — a buffer drawn that way held the same stale
  frame indefinitely. Analysis reads a direct `drawImage` of the element.

The sprite rides the mean of the last `TRACK_WINDOW` peaks (5), not the raw
per-frame peak, so single-pixel hops between equally bright neighbours don't
show. Measured with a dot alternating 20 px every frame: the raw peak swings
20 px, the sprite swings 4 px. A single 400 px jump ramps the sprite over the
window — 280.5 → 360.5 → 440.5 → 520.5 → 600.5 px — landing exactly on the new
highlight. The cost is about two frames of lag at 30 fps; `TRACK_WINDOW` is the
knob, and the crosshair is drawn on the sprite rather than on the raw peak, so
the two never disagree.

## offline-check.html

Runs the sketch with no camera and no model download. Both seams are ordinary
function declarations in `sketch.js` — `openCamera()`, `loadDepthModel()`,
`runDepthModel()` — which are window properties, so the harness just replaces
them: `openCamera` hands back a painted canvas, `runDepthModel` returns a
synthetic tensor in the model's shape (`{ data, dims: [h, w] }`, larger = nearer)
with a white block on black. Nothing reaches into p5's internals: p5 2.0 binds its
globals at start-up, so anything installed on window before then gets clobbered.

Useful for checking the trim, the band maths and the tracking after edits. It is
not a substitute for running it with a camera.

**The three files go together.** `index.html` supplies `window.HF` (the
transformers module), `sketch.js` expects it, and `offline-check.html` stubs this
sketch's seams. Reverting one of them alone produces a sketch that waits 30 s on
a module nobody loads and then reports it.

## Compared to ImageCapture

`../ImageCapture` does the depth cut-out and stops there, with the sliders
positioned by viewport coordinates. This one adds the sprite bake and the
brightest-pixel tracker, and lays the controls out in normal document flow under
the canvas so nothing overlaps it.
