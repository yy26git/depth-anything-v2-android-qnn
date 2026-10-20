# Depth Anything V2 Android QNN

Version **0.1.0** is an experimental Android app that runs Depth Anything V2 Small through ONNX Runtime’s Qualcomm QNN Execution Provider on Hexagon HTP and overlays relative depth on a live CameraX preview.

The validated device was a OnePlus 9 with Snapdragon 888, HTP V68 and Android 14. In strict mode, a completed ONNX Runtime profile showed one QNN partition and zero CPU nodes. Eight reference images passed with minimum Pearson correlation 0.99896 and maximum normalized MAE 0.00671. The live camera delivered about 3.25 depth updates per second in the measured session. These results apply to that tested build and device. They do not demonstrate metric depth, ARCore Depth API equivalence, 10 Hz operation, or performance on other devices.

## Implementation

The app uses CameraX preview and analysis with a latest-frame drop policy, an UINT16 W8A16 model interface, and bulk JNI tensor conversion. It reports actual inference and frame-drop counts without smoothing.

The QNN fix is a narrowly scoped ONNX Runtime 1.30 patch. ORT’s QDQ selector had rejected linear Resize groups to protect CPU quantization optimization. QNN reused that selector, so its builder received float Resize inputs and rejected them. A default-off selector option is enabled only for QNN quantized-node grouping; the original five bilinear Resize operations and model graph stay intact. See `patches/ort-1.30-qnn-linear-resize-grouping.patch`.

## Build requirements

- JDK 17, Android SDK platform 35, Android NDK 27.2.12479018, and CMake 3.22.1.
- A compatible Qualcomm Android device for HTP validation. The verified target was Snapdragon 888 / HTP V68.
- Locally provisioned model and runtime files. They are excluded from Git; expected versions and SHA-256 hashes are recorded in `testdata/runtime-uint16.json`.

Place the model, custom ONNX Runtime 1.30 QNN AAR, matching native library and headers, QAIRT 2.50 HTP libraries, and any other files listed by the manifest at their listed paths. Then run:

```bash
python3 scripts/verify-runtime-artifacts.py
bash scripts/verify-project-sources.sh
./gradlew :app:testDebugUnitTest :app:assembleDebug
```

Qualcomm SDK files and model weights may have distribution restrictions. Confirm their terms before redistributing them.

## Install and run

```bash
scripts/deploy-android.sh
```

Set `ADB_SERIAL` if more than one Android device is connected. Instrumentation tests and strict placement checks require a physical compatible Qualcomm phone.

## License

No project-level license has been declared. Check model, SDK, dependency, and fixture rights before publishing or redistributing this code.
