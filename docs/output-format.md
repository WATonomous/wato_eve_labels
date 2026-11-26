# What the model outputs, exactly

## The network is 2D, in image space

Input: one camera image, 3 x H x W (pano cameras are 1280 x 1024, resized to
320 x 256 in the encoder for speed).

Output: a per-pixel class distribution over four classes at image resolution.

```
drivable | walkway | crossing | other      (softmax 4 x H x W)
```

Nothing about this stage is BEV. It is a plain SegFormer-B0 semantic
segmentation model.

## The BEV grid is geometric, downstream

```
for each of the 200 x 200 cells at 0.5 m in the ego frame:
    p = (x, y, z_ground)          # z from the LiDAR ground surface
    for each of the 8 cameras:
        project p with that camera's K and extrinsics
        bilinearly sample the camera's 4-class probabilities
    average the probabilities over the cameras that see the cell
-> 200 x 200 x 4 probability grid in ego BEV
```

So the system output is a **200 x 200 x 4 probability map**, argmaxed into
per-class masks. Identical output shape to the old BEVFusion map head
(200 x 200 x 6), but produced by projection instead of a learned fused BEV
feature.

## Consequences of this design

- **Cost**: about 3 GFLOPs for 8 cameras, ~1 ms on a 3090, versus 23.6 ms for
  the fused BEVFusion stack.
- **The BEV has no memory.** Each cell's value comes from one or more camera
  pixels in that frame only. A cell hidden behind a parked car is still assigned
  a class unless an occlusion test removes it — which is not implemented yet.
- **Radial structure in the renders is camera FOV, not the scene.** Where wedge
  boundaries appear in the BEV panels, that is the edge of a camera's field of
  view; the black wedges are "other" (sky), i.e. genuinely unknown.
- **Range is limited by resolution.** Beyond ~40 m a 0.5 m cell covers only a
  few pixels, so the BEV degrades with distance for geometric, not learned,
  reasons.
