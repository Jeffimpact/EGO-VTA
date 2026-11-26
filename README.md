# EGO-VTA

EGO-VTA (Egocentric Video-to-Action) measures hand motion and pinch aperture in manipulation recordings. Start with one MP4 and the CPU tracker. A separate calibrated workflow supports tracked cameras, sensor data and policy-format exports.

![Four-second synthetic hand with synchronized skeleton and pinch aperture](docs/synthetic-preview.gif)

The preview uses known synthetic landmarks to illustrate the output. The public-video measurements in the [build plan](BUILD_PLAN.md) use the CPU hand detector.

## Start here with one phone video

The first milestone runs MediaPipe on CPU. Install Python 3.12 and the tracking extra in a virtual environment. No PyTorch, GPU, camera driver, or paid API is needed for this command.

```sh
python -m pip install -e '.[tracking]'
python -m src.pipeline.run --input my_clip.mp4 --seconds 15 --output-dir data/exports/first --gif
```

Substitute any local MP4 for the input. A 15-second desk view of a hand pinching and lifting a pen is a suitable controlled test. Start with the phone stationary. Use a head- or chest-mounted phone for a separate moving-camera trial. The first run retrieves Google's pinned 7.8 MB hand model and verifies its checksum. To run offline, pass a pre-downloaded bundle with `--model path/to/hand_landmarker.task`. On Linux install the EGL and GLES runtime libraries if the MediaPipe wheel requests them.

**Outputs.** `diagnostic.mp4` shows the video with 21 joints, an oblique view of the hand-local 3-D skeleton and its bounding box, and a pinch-aperture curve with a synchronized cursor. `episode.h5` contains the observation and proxy-action tensors when the clip meets the quality gates. `qc.json` records every gap and each valid interval. A rejected clip keeps its diagnostic and QC report, returns exit status 3, and emits no accepted tensor file. The optional `--gif` writes up to ten seconds for a portfolio. Use your own footage for the public example.

| Dataset in the clip HDF5 | Contract |
| --- | --- |
| `images` | RGB uint8, `(T,224,224,3)` |
| `timestamp_ns`, `source_timestamp_ns`, `source_frame_index` | Uniform 20 Hz target clock, actual exposure and original decoded frame |
| `hand/landmarks_uv`, `hand/landmarks_hand_centered_m` | 21 image landmarks and the model's hand-centered 3-D estimate |
| `state` | `(T,4)`. For plain RGB the fields are wrist u, wrist v, projected palm span in image heights, and normalized pinch aperture |
| `actions` | `(T,4)` human proxies. Deltas of the first three state fields and next-frame aperture. Invalid transitions, including the terminal row, contain NaN |
| `valid/landmarks`, `valid/action` | Boolean masks for each time step. Missing hands are never filled with zero-valued observations |
| `synthetic/imu`, `synthetic/fmg` | Present with `--mock-sensors`. Raw 100 Hz samples, aligned values, age, and synthetic provenance. Excluded from grading |

Aperture uses 3-D thumb–index distance divided by palm width. Use `--pinch-range CLOSED OPEN` to calibrate the [0,1] scale with an open-hand and pinch recording. The defaults are illustrative. Single-camera RGB has no metric wrist-depth label in this profile. A camera calibration JSON with `K`, `D`, `model`, `image_size`, and measured `palm_width_m` enables estimated camera-frame XYZ via PnP. Head motion still affects that frame. The calibrated pipeline described below can additionally use a tracked camera pose.

**Reproduce the real-video test without a camera.** The helper fetches individual unmodified MP4s and paired ARKit reference files from Apple's archive using HTTP byte ranges. The default two episodes are under 30 MB of video. The sticker clip supplies a sustained hand track; the untrimmed pour clip exercises rejection.

```sh
python tools/fetch_egodex.py data/raw
python -m src.pipeline.run --input data/raw/sample_peel_place_sticker.mp4 --seconds 15 --output-dir data/exports/sticker --mock-sensors --gif
```

Run `python tools/check_pinch.py data/exports/sticker/episode.h5 data/raw/sample_peel_place_sticker.hdf5` to compare the raw aperture ratio against the paired ARKit reference. `--start-second` and `--seconds` select a window on the original video clock.

Use `--hand Left` when the left hand is the subject. The strict whole-clip gate rejects over 5% invalid observations or a gap longer than five 20 Hz frames. An optional `--pitch-csv` with calibrated `timestamp_ns,pitch_deg` enables the 35° pitch gate. With video alone, the report names the observed tracking loss and marks pitch unmeasured. See [the build plan and benchmark results](BUILD_PLAN.md).

## Run in a cloud checkout

From the repository root in a Python 3.12 environment, run the CPU checks below. The smoke run downloads two original EgoDex clips and the pinned model on first use. It checks the 15-second tensor and diagnostic video, the independent aperture comparison, and a rejected pour episode. Outputs are written under `data/exports/cloud/`.

```sh
python -m pip install -e '.[tracking]'
python -m unittest discover -s tests -v
python tools/cloud_smoke.py
```

For a Codex environment with offline execution, use the install command plus the following two commands in its network-enabled setup phase. The test and smoke commands then use the local files. A minimal Linux image needs `libgl1`, `libegl1` and `libgles2` for the tracker.

```sh
python tools/fetch_egodex.py data/raw
python -c 'from ego_action.clip import model_path; print(model_path(None))'
```

## Generate the offline example

Run `ego-vta demo sample` after installation to generate a four-second synthetic recording under `sample/capture` and its processed outputs under `sample/result`. Use `ego-vta demo sample --formats hdf5,mcap,lerobot` in a fresh directory with the LeRobot extra to exercise all three formats. The virtual camera moves independently of the hand. Open `sample/result/review.mp4` to see the recording and `sample/result/episode.json` to inspect the quality decision. `ground_truth.csv` is an independent test oracle and is never read by the pipeline.

## Install and run

Use Python 3.12 for all three exporters. The core supports Python 3.10 and later. On a Linux laptop, the following commands select CPU PyTorch wheels. The initial install needs internet access.

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --torch-backend cpu -e '.[tracking,lerobot,dashboard]'
ego-vta demo sample --formats hdf5,mcap,lerobot
ego-vta inspect sample/result
```

For a smaller install, `python -m pip install -e .` installs the HDF5 and MCAP pipeline. Run `ego-vta demo work/good` with those defaults. The dashboard, detector, and LeRobot each have their own extra. The tracking wheel on Linux uses the system EGL and GLES libraries even for CPU inference. The Dockerfile installs those libraries.

| Command | Result |
| --- | --- |
| `ego-vta demo work/lost --fault occlusion` | A controlled tracking failure, a QC report, and a quarantine decision |
| `ego-vta demo work/force --fault force` | A visually closed grasp with no FMG activation, quarantined with a reason |
| `ego-vta run capture/episode.json work/run-01 --formats hdf5,mcap,lerobot` | Process a capture manifest and write selected formats |
| `ego-vta inspect work/run-01 --html review.html` | Single-file offline inspector with video, scrubber, 3-D wrist path, aperture, speed and segment text |
| `python -m unittest discover -s tests -v` | Check temporal alignment, known motion, file readers, failure cases, staging, and the local VLM contract |

The dashboard opens in a local browser. The offline HTML also works when the source recording and backend are elsewhere. Its size includes a copy of the review video.

## Use your own recording

```sh
ego-vta init-video recording.mp4 capture.json --task "Move the block into the tray" --model hand_landmarker.task
ego-vta run capture.json work/first-clip
```

For a first glasses clip, choose the hand in view with `--hand Right` or `--hand Left`. The initial manifest uses a moving-camera profile and records a preview. To produce metric actions, fill in the intrinsics, measured palm width, closed/open fingertip distances, and a fixed or tracked camera reference. For a fixed camera trial, set `reference` to `static_camera` only when the camera was stationary throughout the trial. A head-mounted recording supports a synchronized camera-pose CSV. A calibrated external landmark tracker can supply the 21 points directly. The generated aperture values are examples to replace with operator measurements. Google publishes the hand model bundle and API guide. [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)

The input contract, channel names, transforms, sensor units, and quality defaults are in [docs/INPUTS.md](docs/INPUTS.md). The smallest trial can use a phone or glasses recording for visual inspection. Physical units and a stable frame of reference are required for the training route.

## Read the exported data

```python
import h5py
from lerobot.datasets import LeRobotDataset
from pathlib import Path

tier = Path("sample/result/synthetic/tier-1").resolve()
with h5py.File(tier / "episodes.h5") as f:
    ep = f["data/episode_0000"]
    images = ep["observation/image_primary"]
    actions = ep["action/chunk"]
    print(images.shape, actions.shape)   # (76, 3, 224, 224), (76, 4, 7)

ds = LeRobotDataset("local/ego_action", root=tier / "lerobot", video_backend="pyav")
frame = ds[0]
print(frame["task"], frame["action"], frame["observation.images.primary"].shape)
```

For world-model targets, `target/future_observation_index` addresses `sequence/image_primary`, including the final horizon of frames. The HDF5 anchor image array is a same-file virtual view over those pixels. The canonical HDF5 file retains per-camera exposure times, source indices, optional telemetry validity, high-level task tokens, and segment IDs. LeRobot v3 is written by the official writer, including its MP4, Parquet, metadata, statistics, and task tables. Each contiguous span becomes a separate LeRobot episode. The ordinary `action` field is seven-dimensional. `action_chunk` stores the full anchor-relative horizon. ROS 2 MCAP includes CDR-encoded RGB images, poses, instructions, measured telemetry, and both one-step and chunked actions. A ROS installation is optional for writing or reading MCAP in Python. [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) · [MCAP ROS 2 writer](https://mcap.dev/docs/python/mcap-ros2-apidoc/mcap_ros2.writer)

## Labels, staging, and edge feedback

The default automatic labeler proposes transitions from grasp changes and speed minima. Its generic captions identify motion and hand opening or closing. A local multimodal server can name objects and write imperative instructions. Set a model and loopback endpoint in the manifest, then remove the optional `annotations` entry to select automatic labeling. Each segment sends three ordered RGB keyframes to that server. The default request uses structured JSON output. Set `structured_output` to `false` for servers that support plain chat completions only. Imported human-reviewed annotations take precedence when supplied.

```json
{"vlm": {"url": "http://127.0.0.1:8080/v1/chat/completions", "model": "local-vlm", "timeout_s": 90}}
```

The CLI can stage immutable captures and retrieve them for processing. A versioned manifest appears after all input assets finish uploading. Retrieval checks each SHA-256 digest. Use an S3 URI with your own configured credentials and the `s3` extra, or use a local directory. These commands run the same staging flow entirely on disk.

```sh
uri=$(ego-vta stage sample/capture/episode.json work/staging)
ego-vta fetch "$uri" work/downloaded
ego-vta run work/downloaded/episode.json work/from-staging
```

For live rig feedback, the capture driver can pipe JSON lines to `ego-vta monitor`. Each line carries integer `timestamp_ns` and calibrated `pitch_deg`, measured `lux`, or both. It emits active and cleared events when pitch drift exceeds 35° or measured illumination falls below 15 lux. Hardware acquisition drivers remain rig-specific.

## Reproducibility

Each successful run contains the source hashes, selected exports and software version in `receipt.json`. An identical request reuses the result. Changed inputs require a new output directory. Failed processing leaves the input capture intact and publishes no partial dataset. Synthetic outputs use a distinct directory, and a quarantined recording produces review material and zero training rows.

For a containerized run from this directory:

```sh
docker build -t ego-vta .
docker run --rm -v "$PWD:/data" ego-vta demo /data/work/container-demo
```

See [BUILD_PLAN.md](BUILD_PLAN.md) for the implementation milestones, acceptance results, and the hardware decisions for a real pilot.
