# MediaPipe Body Pose Sender

This Python sender captures body pose landmarks with MediaPipe and sends each pose frame through two outputs:

- Windows Named Pipe for the existing Unity project.
- UDP Protobuf for Android or other future clients.

The Named Pipe path is kept for compatibility. The UDP path is an additional output and does not replace the original Unity pipe protocol.

## Run

```powershell
uv run python .\main.py
```

If you are not using `uv`, install the dependencies from `pyproject.toml` in a Python 3.10 environment and run:

```powershell
python .\main.py
```

## Configuration

Edit [global_vars.py](global_vars.py) to change camera, smoothing, debug, and UDP settings.

```python
UDP_ENABLED = True
UDP_TARGET_IP = "127.0.0.1"
UDP_TARGET_PORT = 9999
```

For Android testing on the same Wi-Fi network, set `UDP_TARGET_IP` to the Android device IP address and make sure the Android app listens on `UDP_TARGET_PORT`.

## Existing Unity Named Pipe Output

The sender still connects to the existing Windows Named Pipe:

```text
\\.\pipe\UnityMediaPipeBody
```

Each frame is sent as:

```text
4-byte unsigned integer payload length + UTF-8 text payload
```

The text payload contains two landmark sets:

```text
FREE|index|x|y|z
ANCHORED|index|x|y|z
```

Each valid pose frame contains 33 `FREE` landmarks and 33 `ANCHORED` landmarks.

## UDP Protobuf Output

Each valid pose frame is also sent through UDP as a Protobuf binary payload.

The schema is defined in [pose_frame.proto](pose_frame.proto):

```proto
syntax = "proto3";

package mediapipebody;

message PoseFrame {
  uint64 frame_index = 1;
  uint64 timestamp_ms = 2;
  repeated LandmarkSet landmark_sets = 3;
}

message LandmarkSet {
  LandmarkSetType type = 1;
  repeated Landmark landmarks = 2;
}

message Landmark {
  uint32 index = 1;
  float x = 2;
  float y = 3;
  float z = 4;
}

enum LandmarkSetType {
  FREE = 0;
  ANCHORED = 1;
}
```

The Python sender currently encodes this Protobuf wire format directly in [body.py](body.py), so no generated Python Protobuf file is required on the sender side.

## Android Integration Notes

Copy `pose_frame.proto` into the Android project and generate Java or Kotlin Protobuf classes from it. Then listen for UDP packets on the configured port and parse each received datagram as `PoseFrame`.

Important details:

- UDP payload is one complete `PoseFrame`.
- No extra length prefix is added to UDP packets.
- `frame_index` increments once per valid pose frame.
- `timestamp_ms` is Unix time in milliseconds from the Python sender.
- `landmark_sets.type == FREE` contains the computed world-space landmarks.
- `landmark_sets.type == ANCHORED` contains the MediaPipe anchored world landmarks.

## Prompt For AI Android Integration

You can paste this prompt into another AI assistant when working on the Android side:

```text
I have a Python MediaPipe body pose sender that sends UDP datagrams encoded as Protobuf. I copied `pose_frame.proto` into my Android project. Please help me integrate it.

Requirements:
1. Generate Android Java/Kotlin Protobuf classes from `pose_frame.proto`.
2. Start a UDP receiver that listens on port 9999.
3. Each UDP datagram is exactly one serialized `mediapipebody.PoseFrame`; there is no length prefix.
4. Parse each datagram with the generated `PoseFrame` class.
5. The schema contains:
   - PoseFrame: uint64 frame_index = 1, uint64 timestamp_ms = 2, repeated LandmarkSet landmark_sets = 3
   - LandmarkSet: LandmarkSetType type = 1, repeated Landmark landmarks = 2
   - Landmark: uint32 index = 1, float x = 2, float y = 3, float z = 4
   - LandmarkSetType: FREE = 0, ANCHORED = 1
6. Use FREE and ANCHORED landmark sets separately. Each set should contain 33 landmarks.
7. Keep UDP receiving on a background thread/coroutine and dispatch parsed pose data to the render/update layer safely.

Please show the Gradle Protobuf setup, the UDP receiver code, and an example of reading the FREE and ANCHORED landmarks from a parsed PoseFrame.
```
