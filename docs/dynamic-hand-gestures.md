# Dynamic hand gesture foundation

This stage adds a dynamic hand sequence format and local sequence classifier in
`gesture_manager.dynamic_gestures`. It is a separate representation from the
static pose data and SVC model. It supports one visible hand per sequence.
Static pose datasets, model files, and recognition are unchanged.

## Recording and feature data

`DynamicRecordingSession` accepts fresh `DetectedHand` frames between
`begin_example()` and `finish_example()`. It rejects multi-hand or malformed
frames, ignores duplicate frame sequence numbers, caps a take at 60 frames,
and only commits completed takes with at least 8 frames. `cancel_example()`
discards the in-progress take while preserving completed takes.

Each frame stores 66 numeric values: the existing 63-value normalized hand
pose plus the wrist's x/y/z motion from the first frame in units of the first
frame's palm scale. This retains motion direction while reducing sensitivity
to camera position and hand size. The store writes JSON feature sequences only;
it never accepts or writes camera images or video. Dynamic datasets use
`gesture-manager-dynamic-dataset`, format version 1, and feature version
`hand-motion-sequence-v1`. Each dataset contains `id`, `name`, `hand_count`,
`gesture_type: "dynamic_hand"`, and `examples`; each example contains an ID,
UTC capture timestamp, and `frames`, an array of 8–60 arrays of 66 finite
numbers. A take is capped at 60 frames and a gesture at 500 takes.
`DynamicGestureStore` should be given a separate directory from the static
`GestureStore`.

## Training and cautious testing

`DynamicGestureTrainer` uses nearest-example dynamic time warping (DTW), which
allows examples to vary in speed without fitting a large sequence model to a
small personal dataset. It groups near-duplicate resampled sequences before a
fixed-seed grouped holdout: a 12-frame embedding and RMS radius 0.025 identify
near-duplicate groups, then `GroupShuffleSplit` uses a 25% holdout with seed
2026 and requires every label on both sides. The evaluation report includes class counts,
independent group counts, accuracy, confusion matrix, and warnings. Results
are marked unreliable until every gesture has at least five examples and five
independent sequence groups. Models are data-only JSON under the separate
`gesture-manager-dynamic-dtw-model` format, version 2, and can be reloaded
without retraining. The model stores the default maximum DTW distance (0.35)
and minimum separation (0.015); `with_thresholds()` changes these values
without retraining, and the changed values survive save/reload.

Examples do not currently store a recording-session ID. The split prevents
near-duplicate sequences from crossing partitions, but cannot guarantee that
different takes from the same longer recording session stay together. Collect
and validate each take independently until session metadata is added.

The model's distance and separation are not probabilities. `predict()` uses
the saved limits by default and can accept one-off `max_distance` and
`minimum_separation` overrides. Choose persistent limits from fresh examples
that were not used for training, and treat a result as uncertain when distance
exceeds its maximum or the best-vs-next distance gap is too small. Do not use
training accuracy to set these limits. The current accuracy and calibration
tests use synthetic sequences only; thresholds have not been calibrated on
webcam recordings.

## Calibration guidance

For each custom motion, record the same start and end pose, keep the hand fully
visible, and perform the movement in a clear direction. Record several takes
with natural speed and small changes in starting position and distance. Use a
plain background when possible. Review confusing pairs with fresh takes and
increase distance/separation strictness if they overlap. Do not record a take
with another hand entering the frame.

## Gesture Studio workflow

Gesture Studio offers a separate **Dynamic hand sequence (experimental)**
choice. Select an existing label to add takes or enter a new label name. Start
the local camera, start a take, and stop/save after at least 8 fresh landmark
frames (maximum 60). A camera error, page departure, or explicit cancellation
discards only the unfinished take. Completed takes are saved immediately as
numeric sequences. Dynamic datasets and the optional evaluated model are stored
per profile under `dynamic_gestures/`, separate from static pose data and models.
Profile export/import does not yet bundle these sidecar assets.

The Studio consumes the existing hand engine's processed landmark snapshots;
it does not open a second camera or run another landmark model. Evaluation runs
the existing grouped DTW trainer in a worker thread. The UI reports per-label
precision/recall and a confusion matrix only when the trainer formed a valid
holdout containing every label. If that grouped split cannot be formed, no
accuracy is shown and the result explicitly says evaluation is unavailable.
Metrics are not calibrated to webcam use, and the grouping logic cannot ensure
that all takes from a recording session remain together because session IDs are
not part of the current dataset format.

Dynamic sequences remain experimental and are not connected to live runtime
recognition, the static pose recognizer, or action execution. A user-facing
duration control, optional input methods, head/facial movements, and wink
detection remain later increments. No accessibility-effectiveness claim is
made; that needs testing with users who rely on alternative input.
