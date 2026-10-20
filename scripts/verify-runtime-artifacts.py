#!/usr/bin/env python3
"""Fail builds if native link/package artifacts or the original model drift independently."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'testdata/runtime-uint16.json').read_text())
for relative,expected in m['files'].items():
    path=root/relative
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit(f'Runtime artifact identity mismatch: {relative}')
patch=m.get('ort_source_build',{}).get('patch')
if patch and hashlib.sha256((root/patch['file']).read_bytes()).hexdigest()!=patch['sha256']:
    raise SystemExit('ORT source patch identity mismatch')
aar=zipfile.ZipFile(root/f"app/libs/onnxruntime-android-qnn-{m['ort']}.aar")
for member,local in [('jni/arm64-v8a/libonnxruntime.so','app/src/main/cpp/ort_link/libonnxruntime.so'),('headers/onnxruntime_c_api.h','app/src/main/cpp/ort_include/onnxruntime_c_api.h')]:
    if aar.read(member)!=(root/local).read_bytes():raise SystemExit(f'Native bridge/AAR mismatch: {local}')
fixtures=json.loads((root/'testdata/qnn-device-fixtures.json').read_text())
for fixture in fixtures['fixtures']:
    for kind in ['input','reference']:
        expected=fixture[kind]
        raw=(root/expected['path']).read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise SystemExit(f"Device fixture identity mismatch: {fixture['name']} {kind}")
print(f"Runtime identities verified: ORT {m['ort']}, QAIRT {m['qairt']}, original UINT16 model")
