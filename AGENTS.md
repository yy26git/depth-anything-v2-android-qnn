# Project guidance

Read `README.md` before editing. Keep package names under `com.yy26git.depthanythingv2qnn`.

This is an experimental Android depth-estimation app. Preserve the UINT16 model contract, quantization constants, QNN-only graph-grouping patch behavior, lifecycle ownership, and truthful frame-drop/rate reporting. Do not claim metric depth or ARCore equivalence; see README for measured limits.

Runtime binaries and the main model are locally provisioned and ignored. Keep `testdata/runtime-uint16.json` hashes accurate. A successful session is not proof of HTP placement; require a completed profile with the strict QNN partition gate.
