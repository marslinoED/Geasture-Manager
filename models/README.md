# Hand Landmarker model asset

- File: `hand_landmarker.task` (MediaPipe Hand Landmarker full model bundle, float16, version 1)
- Official download source: <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
- SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- License: Apache License 2.0, as stated by the official MediaPipe Hand Tracking model card: <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf>

The bundle contains the palm detector and hand landmark model required by the MediaPipe Hand Landmarker task. The app must load this local file and must not download it at runtime.

The two `.tflite` files in this directory were extracted from that exact bundle for direct local execution with the pinned AI Edge LiteRT interpreter. The app does not use the MediaPipe Tasks runtime. Image preparation, palm box decoding, and landmark projection are performed locally by `gesture_manager.hand_landmarks` using the model graph configuration from the Apache-2.0 licensed MediaPipe source.

Extracted asset checksums:

- `hand_detector.tflite`: SHA-256 `945f713bc23570bd4ed60f848c401dc8eaf95713183d43ba14cf12e467d27a7d`
- `hand_landmarks_detector.tflite`: SHA-256 `6acda74af3fbf40e68265c20c7394b2bad81a16a481dcd79ad7a081887c3d6b9`

Relevant upstream graph configuration: [palm detection graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt) and [hand landmark graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt). The model card identifies Apache License 2.0; the source project license is [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).
