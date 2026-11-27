# Engineering Specification: Automated Egocentric Video-to-Action Data Pipeline System

## Objective and scope

Given an episode and a feature row, a reviewer should be able to identify the recording, find the decoded frame, see the selected hand, and explain both the feature's units and its eligibility. The first release tests this on one hand, 15 seconds of RGB and a CPU. A second interface handles calibrated, synchronized capture manifests. Both run as batch jobs. Robot-specific retargeting, policy learning and physical task success are later milestones.

The [build plan](../BUILD_PLAN.md) records the original experiments. The [capture contract](INPUTS.md) defines the manifest and coordinate conventions. The [roadmap](ROADMAP.md) defines the next experiments.

## Interfaces and invariants

| Contract | Single-video feature review | Calibrated episode processing |
| --- | --- | --- |
| Input | One MP4, selected hand, model. Optional window, pinch calibration, camera calibration or calibrated pitch log | Manifest with task, primary camera and time origin. Optional additional cameras, landmarks or detector, camera poses, sensor streams and segment annotations |
| Clock | 20 Hz. Time zero is the first decoded video presentation timestamp. Window offsets retain that origin | 10 or 20 Hz. Each video's first presentation timestamp maps to its declared start on the common clock |
| Frame selection | Match grid to nearest primary exposure within tolerance. Carry decoded frame index and source time. Repeated selections count as missing | Align secondary views to primary exposure; required streams participate in eligibility. Default sensor alignment looks backward up to 15 ms |
| Observation | RGB uint8 THWC, normalized wrist u/v, projected palm width in image heights and scaled aperture. Calibration switches the first three state fields to estimated per-exposure camera XYZ in metres | Rectified views when calibrated, wrist pose in the declared reference, normalized aperture, timed telemetry and language segments |
| Target | Three one-step feature differences and the next scaled aperture, shape (T,4). Missing observations invalidate both adjacent transitions. The final action is NaN | Each anchor has H future targets in the same valid span. Each target has three anchor-frame translations (m), three relative rotation-vector components (rad) and the future aperture |
| Output decision | Write diagnostic and QC for accepted and rejected clips. Only accepted clips receive episode.h5 | Write review and provenance for every completed episode. Eligible spans feed selected exporters. Quarantine yields zero training rows |
| Replay | qc.json records the input/model hashes, parameters and artifact hashes | receipt.json records recipe, version, input and output digests. Use a new destination for changed inputs |

Aperture starts with the distance between thumb tip and index tip divided by the index-to-pinky MCP distance in hand-local 3-D. Store that raw ratio for comparison. The clip's scaled aperture maps the configured closed/open ratios to [0,1]. The manifest route uses metric closed/open distances. Hand-centered model coordinates alone cannot establish global wrist translation. For a moving camera, reference-frame motion requires synchronized camera poses. FMG measures activation. Object contact requires independent evidence.

## Processing responsibilities

1. **Decode and align.** Validate timestamps and calibration. Select frames from their presentation times, preserving grid time, exposure time and source index. Invalid selections break continuity. The manifest's offline interpolation mode requires nearby samples on both sides and should be recorded when evaluating causal policies.
2. **Estimate geometry.** Select one handedness. Keep normalized image landmarks and local skeleton separate. For calibrated input, fit scale with the measured palm width, solve PnP and reject implausible fits. Apply the supplied camera-to-reference transform before computing stable-frame deltas.
3. **Assign labels and check evidence.** In the manifest route, imported annotations take precedence. Otherwise motion and aperture suggest boundaries; a configured local VLM may caption selected keyframes. Run quality checks before building action windows. Report missing evidence and keep optional synthetic sensor samples explicitly identified.
4. **Build and serialize targets.** The clip route writes a validity mask plus NaN for each unusable action. The manifest route splits valid spans before smoothing and chunking. A contiguous span of length N supplies max(0, N−H) anchors. HDF5 carries dense arrays; MCAP carries ROS 2 messages; the optional LeRobot writer creates episodes per contiguous span.
5. **Review and reproduce.** Read the quality report alongside the video. For an individual row, use its source index to inspect the original frame. Use the aperture-reference comparison to test measurement agreement independently of whether output serialization succeeded.

## Verification and release evidence

| Requirement | Verification | Acceptance or interpretation |
| --- | --- | --- |
| Trace a row to video | Run tools/cloud_smoke.py. Reopen HDF5, decode the diagnostic and compare the recorded source frame for a selected event | The 15 s sticker window has 300 image and action rows and 300 diagnostic frames. Grid spacing is 50 ms. Inspect the matched source frame when judging event timing |
| Reject tracking loss | Run the same smoke check on the full pour clip; inspect qc.json | For the clip route, reject above 5% missing or a run longer than five rows. Keep the diagnostic and QC. An accepted clip can contain masked rows |
| Keep measurement evidence separate | Compare raw sticker aperture to confidence-filtered paired ARKit joints with tools/check_pinch.py | Record matched count, rank correlation and mean absolute error in palm widths. The original 114 matches gave 0.588 and 0.236 respectively. These are single-window observations |
| Preserve calibration and export semantics | Run the synthetic moving-camera demo and tests/test_pipeline.py | The default clean demo has 80 policy observations and 76 anchors at H=4. Check independently known motion, future-frame indices and reader agreement. Report optional-reader skips |
| Preserve output integrity | Run the replay, conflicting-input and staged round-trip tests | Identical completed requests can be reused. A changed recipe requires a fresh destination. A corrupted staged asset must fail verification |
| Establish usefulness on personal footage | Manually mark pinch onset, lift and release in stationary and moving-phone trials | Pending. Record frame-level errors and tracking failures per view before choosing hardware or a controller |

## Operating boundaries

Use reviewed manifests for staging and check the resolved asset list before uploading captures. A staged bundle's checksums detect changes relative to its manifest; producer identity requires a separate trust decision. An exported HTML inspector embeds a copy of the review video. Share it under the same permissions as the footage. Use a fresh writable results directory and retain the input manifest, raw media and calibration alongside the run record.
