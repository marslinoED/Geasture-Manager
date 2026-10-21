# Custom gesture training and validation

## Training behavior

The trainer reads the versioned local gesture datasets and uses their IDs and names as the model label mapping. A model requires at least two custom labels, three valid examples per label, one consistent hand configuration across labels, and at least two distinct feature vectors in each class. Invalid, non-finite, out-of-range, or incompatible features stop training with a useful error.

The current classifier is an RBF support vector classifier from scikit-learn 1.9.1. It is trained on normalized local hand landmark features. Training and holdout evaluation run in a worker thread. Tkinter receives results through a queue. Live classification has a separate worker and accepts only newly processed camera results.

## Live recognition gate

The live recognition path emits structured `RecognitionEvent` values and remains separate from action execution. The Actions page consumes only trigger-eligible events and dispatches separately configured, allowlisted keyboard shortcuts; it does not connect actions to classifier training. A custom SVC score is computed as the difference between the winning class's aggregated one-vs-one decision evidence and the runner-up's evidence. This is a raw, uncalibrated decision margin, not a probability. Each gesture has an editable minimum margin; the default `0` rejects ties and non-positive margins but is not a calibrated low-confidence threshold. Set a higher value only after checking held-out examples for that gesture. A vote tie is always considered uncertain and shown as **Uncertain — no gesture detected**.

By default the predicted gesture must appear on three consecutive fresh inference frames. After a diagnostic event becomes eligible, that pose cannot become eligible again until two consecutive no-hand frames establish neutral release. A one-second per-gesture cooldown adds another repeat guard. The live panel lets users enable/disable each class and edit its threshold, frame count, and cooldown. Disabled labels are excluded from the SVC pair comparisons, so live testing cannot report them as the selected gesture; fewer than two enabled labels produce an uncertain result. Gesture enablement and margin thresholds apply during the live test; frame count and cooldown edits made during a test apply the next time it starts. These recognition controls are in-memory for the current app session and are not part of the model file. Pause halts recognition and requests camera pause; resuming requires a neutral release. Emergency stop ends recognition and requests camera release. These controls remain separate from future action execution.

## Leakage-aware evaluation

Before splitting, feature rows within RMS distance 0.015 are connected into a shared pose group. The deterministic grouped split keeps a whole group on one side of the holdout, so repeated or nearly identical captures do not appear in both training and validation. A stratified group split is attempted first; a deterministic group shuffle is used if that cannot include every class. If no valid group holdout can be formed, the model can still train on valid user data, but the UI reports that it has no validation metrics.

Displayed accuracy, per-class precision/recall/F1, and the confusion matrix are computed only from the grouped holdout. Training accuracy is not reported. Validation is marked preliminary unless every gesture has at least five examples and five independent pose groups, and at least two holdout examples from two groups. These thresholds describe the available sample size; they do not establish real-world reliability. The live test uses fresh camera samples outside the saved training data and reports labels without confidence claims.

## Model storage and invalidation

The app saves `custom_gestures.json` under `%LOCALAPPDATA%\GestureManager\models`. It contains the feature version and size, custom label mapping, classifier format/version, scikit-learn version, timestamp, dataset fingerprints, sample/group counts, holdout metadata, and portable SVC parameters. Writes use a temporary file followed by atomic replacement. The model is plain validated JSON; the app does not unpickle estimator files or execute profile content.

At startup, the app validates and loads an existing model in the background. It does not retrain automatically. A model is current only when all saved dataset fingerprints match. Saving changed examples, renaming a gesture, or deleting one marks it stale and disables live testing until retraining. Missing, malformed, or incompatible model files are reported safely; training can replace an invalid file.

## Candidate benchmark

The local development benchmark compared a balanced RBF SVC (`C=10`, `gamma=scale`) and a balanced 100-tree random forest (maximum depth 12) on the same synthetic grouped data: 144 samples, 126 features, four generated classes, 48 pose groups, three near-identical captures per group, and eight group-separated holdouts. This is a software/model-selection check, not evidence of gesture quality.

| Candidate | Mean grouped macro F1 | Mean fit time | Mean prediction time |
| --- | ---: | ---: | ---: |
| RBF SVC | 1.0000 | 1.38 ms | 0.300 ms |
| Random forest | 1.0000 | 112.37 ms | 4.767 ms |

Both candidates separated these intentionally distinct synthetic classes. The SVC was selected because its measured training and inference times were lower in this benchmark. These timings and scores do not predict performance on real user gestures. The installed Python 3.13 Windows wheel is available for scikit-learn 1.9.1; see the [official installation guide](https://scikit-learn.org/stable/install.html) and [PyPI release files](https://pypi.org/project/scikit-learn/).

## Manual test

1. Create at least two custom gestures with the same hand count. Record varied examples for each; five or more distinct pose groups per label are recommended.
2. Save each dataset and select **Train and evaluate**. Confirm the status changes while training runs and the window remains responsive.
3. Review the grouped holdout summary, per-gesture metrics, and confusion matrix. Confirm a small dataset carries a preliminary warning and no training accuracy is presented.
4. Start **Live test** and present fresh gestures to the webcam. Confirm the UI displays current custom labels; this is a smoke test and not a validation score.
5. Optionally enable the preview and verify camera landmarks remain local. Stop the live test and confirm camera release.
6. Modify or remove examples and verify the saved model is stale until retrained. Restart with unchanged data and verify the model loads without retraining.
7. Try a missing, malformed, or incompatible model file. Confirm the app reports it and remains usable for retraining.
8. In the live panel, raise one gesture's margin threshold and verify low-margin predictions become uncertain; disable another gesture and verify it never becomes trigger-eligible.
9. Hold a pose and confirm it reports eligible once only after the configured consecutive frames. Confirm it stays ineligible until two no-hand frames, then observe the per-gesture cooldown.
10. Pause, resume, and verify a release is required before eligibility returns. Test **Emergency stop** and confirm the camera release is requested and no further diagnostic trigger events appear.
11. Disable one custom gesture while testing. Confirm the disabled label is not displayed as a live prediction; with fewer than two enabled labels, the panel should report uncertainty.
12. Resize the window to its minimum size. Confirm the page scrollbar reveals all content and the Camera controls remain on separate rows without compressing the preview.
