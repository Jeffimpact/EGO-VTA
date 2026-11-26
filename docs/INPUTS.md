# Capture and action contract

Use UTF-8 JSON and CSV. Paths in the manifest resolve relative to its location. The working example is `../sample/capture/episode.json`. Supply a pseudonymous operator ID if longitudinal operator analysis is needed. One manifest selects one active hand and can include several synchronized cameras.

| Manifest field | Meaning |
| --- | --- |
| `episode_id`, `task` | Simple identifier and the operator's high-level instruction |
| `fps`, `horizon` | A 10 or 20 Hz policy grid, and the number of future steps per anchor. Defaults are 20 Hz and one step |
| `cameras.primary` | `file`, integer `start_time_ns` on the common capture clock, and optional calibration |
| `cameras.<name>` | Additional video with its own start time and calibration. Set `required: false` for a view that can be missing |
| `calibration` inside each camera | `image_size: [width,height]`, `model: "pinhole"` or `"fisheye"`, 3×3 `K`, distortion vector `D`. Calibration applies at the decoded orientation and resolution |
| `reference` | `static_camera` for a fixed optical camera, `tracked_camera` for a moving camera with supplied poses, `moving_untracked` for visual review |
| `camera_pose` | CSV with `timestamp_ns,x,y,z,qw,qx,qy,qz`. Each rigid transform maps the primary optical frame **into** a fixed reference frame |
| `hand_model`, `hand` | Path to the MediaPipe Task model and `Right` or `Left`. The returned handedness category must match. Set `swap_handedness: true` only after checking a mirrored capture |
| `palm_width_m`, `aperture_range_m` | Operator-specific distance between index and pinky MCP joints, and closed/open thumb-to-index fingertip distances, all in metres |
| `landmarks` | Alternative to `hand_model`. CSV with `timestamp_ns,p0_x,p0_y,p0_z,...,p20_z` in primary-camera metres. Use the MediaPipe joint order. Blank values denote missing data |
| `landmark_quality` | Set `validated_metric` for imported landmarks whose scale and reference have been independently checked |
| `sensors` | Map of `imu`, `fmg`, `gaze`, or `rig` to `file`, optional integer `offset_ns`, and `required` (defaults to true for a declared stream) |
| `annotations` | Optional JSON array of segments covering the policy grid exactly once. Uses inclusive `start_frame` and `end_frame`, plus `instruction`. Optional fields include `target_object`, `contact_established`, and `activity` |
| `vlm`, `tokenizer` | Optional local model server settings and a Hugging Face tokenizer JSON file for the target policy. The included fallback is a deterministic byte-level BPE vocabulary |

## Temporal alignment

The decoder reads presentation timestamps with their actual video time base. Primary frames are selected nearest the policy grid (default 25 ms limit). Ties use the earlier frame and repeats are marked missing. Secondary cameras align to the selected primary exposure (default 15 ms). Override a camera's tolerance with `tolerance_ms`. The output preserves both policy-grid time and actual exposure time.

Provide each sensor's time on the common clock or add its known `offset_ns`. Under the default `backward` policy an exposure can use only the most recent finite measurement at or before the exposure, at most 15 ms old. The optional `interpolate` policy is an offline mode. Both surrounding samples must be within 15 ms. Quaternion orientation uses SLERP. FMG uses a local cubic fit when four nearby samples exist, otherwise linear interpolation. The pipeline never extrapolates across a dropout. Clock drift correction and absolute clock calibration belong to capture preparation.

| CSV | Fields following timestamp_ns | Units or range |
| --- | --- | --- |
| IMU | `ax,ay,az,gx,gy,gz` | m/s² and rad/s in the sensor frame. Add `qw,qx,qy,qz` and set `orientation: true` for orientations |
| FMG | `fmg0,fmg1,fmg2,fmg3,fmg4,fmg5` | Session-calibrated [0, 1] muscle activation on six channels. Preserve the original voltage logs separately |
| Gaze | `u,v` | [0, 1] in primary image coordinates |
| Rig | `pitch_deg,lux` | Pitch deviation relative to a neutral calibrated rig and independently measured illuminance |

## Geometry and targets

Images use OpenCV optical axes (+x right, +y down, +z forward). The wrist is landmark 0. Its x-axis points from pinky MCP to index MCP. The middle-finger direction defines the palm plane, and a cross product completes a right-handed wrist frame. Quaternions in files use **w, x, y, z**. ROS pose messages map those components to their named fields.

The hand-centered monocular model estimates shape. With calibrated rectified intrinsics and a measured palm width, PnP fits that shape to the observed pixels and estimates the camera-space wrist. Fits with negative depth or excessive reprojection error (default 8 px) are discarded. A tracked-camera pose moves the whole skeleton into the fixed reference. The pose estimate remains subject to monocular bias. Independent hand-size and camera calibration matter. [MediaPipe landmark coordinates](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)

For anchor t and horizon step k, translation is `R_t.T @ (p[t+k] - p[t])`. Rotation is the rotation vector of `R_t.T @ R[t+k]` in radians. These are finite relative motions. Grasp is the next normalized thumb-index distance, with 0 closed and 1 open. Every target uses only a complete future within the same valid span. A span of N observations and horizon H produces N−H anchors. Smoothing fits against exposure time and resets at every discontinuity. The default backward mode uses only current and earlier poses for the filtered observation.

## Quality decisions

| Evidence | Handling |
| --- | --- |
| Hand or required data missing | Remove affected frames. Flag tracking runs longer than 0.25 s and quarantine above 5% missing hand or required-stream coverage |
| Optical framing | Quarantine on absolute rig pitch above 35°. Use measured lux below 15 when available. An image proxy flags low 95th-percentile brightness or extensive clipping. Quarantine when over 5% of frames fail lighting |
| Dynamics | Excise frames where raw speed exceeds 1.8 m/s or raw acceleration exceeds 15 m/s², then split action windows across the resulting gaps |
| Grasp consistency | Quarantine on normalized aperture ≤0.10 while every calibrated FMG channel is ≤0.02. Report closure–activation Pearson correlation when a dynamic range is present |
| Behavioral purity | Remove labeled idle/personal activity and low-speed runs longer than 3 s. Explicitly annotated holds are retained |
| Confidence | 35% visibility, 25% smoothness, 20% language, 20% FMG. Tier 1 starts at 90 with complete evidence, Tier 2 at 75. Hard failures override the weighted score |

Tier 1 requires complete measured geometry and capture evidence, including reviewed captions, usable rig measurements and FMG agreement. Automatically generated captions and monocular PnP estimates can enter Tier 2 when they pass the other checks. Missing force evidence contributes zero to the force component. An untracked moving camera or missing metric geometry gives a review-only episode. The component values and thresholds support triage. Their predictive accuracy must be measured on independently labeled recordings.

Each export carries source frame indices. Optional sensor gaps in HDF5 retain NaN plus validity flags. LeRobot uses zeros accompanied by an explicit validity channel. MCAP omits missing sensor messages. Grasp and FMG supply evidence about closure and muscle activation. Explicit object-contact labels can come from reviewed annotations.
