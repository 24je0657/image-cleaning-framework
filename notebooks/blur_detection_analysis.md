# Blur Detection Analysis

## Finding
Visual inspection revealed that some cat and dog images
removed by the blur detector (41 cat, 58 dog) appear
visually acceptable to human observers.

## Root cause
Laplacian variance measures **edge density**, not perceptual sharpness.

Low blur scores in cat/dog images occur due to:
- Smooth animal coats (low texture → few edges)
- Plain/solid backgrounds
- Flat, even lighting
- Close-cropped face shots with minimal background

These images have few edges NOT because they are blurry
but because their content is inherently smooth.

## Quantitative evidence
All blur-only removals scored below 50 regardless of class.
Class thresholds range from 65.8 (dog) to 535.4 (lion).
Gap between max removed score and threshold: 16–145 points.
This confirms the threshold is not cutting into normally-textured images.

## Impact
~99 images (1.8% of cat+dog) may be low-texture but acceptable.
These represent a known false positive rate in the blur detector.

## Better alternatives
- BRISQUE: perceptual quality metric trained on human judgments
- NIQE: natural image quality evaluator (no reference)
- ML blur classifier trained on human-labeled data

## Decision
Accepted as documented limitation for current version.
Option C (blur_severity tiering: severe→REMOVE, moderate→REVIEW)
identified as the recommended fix for production deployment.