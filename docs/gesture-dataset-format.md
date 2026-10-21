# Gesture dataset format, version 1

Gesture datasets are stored in the current user's local application data folder:

```text
%LOCALAPPDATA%/GestureManager/gestures/<gesture-uuid>.json
```

Each UTF-8 JSON file contains one custom static-pose gesture. Datasets use the
`gesture-manager-dataset` format identifier and `format_version: 1`. Their
`feature_extraction_version` is `hand-pose-normalized-v1`; a future incompatible
feature extractor must use a new version instead of silently changing old data.

A valid one-hand example uses 63 values in the `features` array. The illustrative values below are all zeroes for compactness:

```json
{
  "format": "gesture-manager-dataset",
  "format_version": 1,
  "feature_extraction_version": "hand-pose-normalized-v1",
  "id": "de305d54-75b4-431b-adb2-eb6b9e546014",
  "name": "My custom pose",
  "gesture_type": "static_pose",
  "hand_count": 1,
  "examples": [
    {
      "id": "de305d54-75b4-431b-adb2-eb6b9e546015",
      "captured_at": "2026-10-21T12:00:00+00:00",
      "features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  ]
}
```

The actual `features` array contains 63 finite numbers for one hand (21 x/y/z
landmarks), or 126 numbers for two hands. Features are translated relative to
the wrist or midpoint between wrists, scaled by average wrist-to-middle-MCP
distance, and rotated to align the average wrist-to-middle-MCP direction. For
two hands, hands are sorted by wrist x coordinate before flattening. This
reduces sensitivity to image position, hand size, and in-plane rotation. It
does not normalize finger articulation or guarantee classifier accuracy.

Only these normalized numeric vectors, IDs, names, versions, hand count, and
capture timestamps are persisted. Camera frames and video are never included.
The reader rejects unknown format/feature versions, unsupported gesture types,
invalid IDs, duplicate example IDs, wrong vector lengths, non-finite or
out-of-range values, invalid timestamps, and unsupported hand counts. Writes
use a temporary file in the dataset directory followed by atomic replacement.
Deletion accepts only a UUID and removes only its corresponding JSON file.

Datasets with fewer than three valid examples are marked unready for training.
This phase does not contain a classifier or a training action.
