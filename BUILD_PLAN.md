# EGO-VTA build plan

The first release answers one question: can a reviewer trace a measured hand feature back to the video frame that produced it? Run the tracker on CPU, starting with 15 seconds and one hand. A synchronized visual and a reference comparison test two different things. The visual reveals missed hands and timing errors. The comparison measures how well a feature follows independent labels.

## 1. Establish a hardware-free baseline

1. Use the original EgoDex sticker and pour clips for reproducibility. The fetch helper retrieves each MP4 and paired reference HDF5 separately from Apple's test archive. Record a 15-second fixed-phone reach, pen pinch, lift, place and release for a shareable demo. A second chest- or head-mounted take introduces real camera motion.
2. Decode original presentation timestamps. At 20 Hz, retain one row per 50 ms, the original source frame index, exposure timestamp and a validity mask. Resampling 30 Hz video is intentional. Missing tracking breaks a transition. An accepted clip can still contain explicitly masked rows.
3. With uncalibrated RGB, export image-space wrist coordinates, apparent palm width in image-height units, and thumb–index aperture divided by palm width. Store the hand-local 3-D skeleton separately. Apparent size also depends on hand orientation. Estimating metric camera-frame translation requires calibration and a scale reference. Comparing wrist motion across a moving camera additionally requires camera poses.
4. Export the named state, one-step feature changes and next aperture, together with the original timing and masks. Produce a synchronized skeleton, local 3-D bounding box and aperture curve. Keep mock IMU/FMG samples opt-in and labeled. They check array shape and alignment. Contact, force, and camera pitch each need their own evidence.

**Acceptance.** Reopen the HDF5 with h5py, verify tensor dimensions and target timestamps, and decode the diagnostic MP4. Inspect onset, lift and release against manually marked frames on the phone clip. For a deliberately clean demo, require at least 95% valid selected-hand observations and at most five consecutive missing rows. These initial gates apply to the chosen window. Preserve valid spans and report the longest gap for natural recordings. A measured pitch excursion can trigger its own rejection. A plain RGB clip reports visibility and marks pitch unmeasured.

## 2. Check the measurements on real clips

These seven selected EgoDex test recordings use episode 0 and the right hand. The first two use the first 15 seconds. The remaining five use the complete recording. These observations describe this sample under the default gates.

| Task | Valid / total rows | Longest gap at 20 Hz | Whole-clip decision |
| --- | ---: | ---: | --- |
| Peel and place sticker | 295 / 300 | 3 frames | Accept |
| Type on keyboard | 292 / 300 | 8 frames | Reject |
| Remove or add lid | 54 / 63 | 9 frames | Reject |
| Pick and place | 55 / 84 | 5 frames | Reject |
| Pour | 90 / 127 | 36 frames | Reject |
| Screw or unscrew bottle cap | 80 / 89 | 3 frames | Reject |
| Stack or unstack cups | 92 / 123 | 21 frames | Reject |

On the sticker clip, 291 adjacent transitions were valid. The extracted raw aperture ratio and the ARKit reference had a Spearman rank correlation of **0.588** across 114 matched frames. Mean absolute error was **0.236 palm widths**. The comparison used frames where all four reference joints had confidence of at least 0.5. Both are estimates from tracking systems. The result motivates checking closure timing across several gestures and views before training a policy. Reproduce the comparison with `tools/check_pinch.py`.

The Linux CPU cloud run passed all 22 automated tests with the optional LeRobot reader installed. A fresh tracking-only install also ran both videos and passed 21 tests, skipping the optional reader. Both runs reopened the sticker tensors and decoded all 300 frames of the diagnostic video. The documentation includes the one-command reproduction after installation.

## 3. Decide what to build next

| Decision | Evidence to collect first |
| --- | --- |
| Add depth or camera pose | Repeat the same motion with a stationary and a moving phone. Compare calibrated metric wrist estimates against a known displacement or independently tracked poses. |
| Add IMU or FMG | Choose a failure the signal should resolve, such as head rotation or an ambiguous grasp. Compare measured, synchronized samples with the video baseline. |
| Start simulation | Select a robot, task and controller. Specify coordinate frames, action units, gripper convention, and how human motion maps to robot commands. Replay a short trajectory before training. |

The repository includes a calibrated pipeline with synthetic fixtures for HDF5, MCAP and native LeRobot, plus the simpler real-video workflow. The benchmark covers CPU batch processing. Target-laptop speed, personal phone footage, physical sensors, robot retargeting and task success remain separate validations.

[MediaPipe landmark coordinate frames](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) · [EgoDex data and reference convention](https://github.com/apple/ml-egodex) · [LeRobot dataset format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
