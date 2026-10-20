#!/usr/bin/env python3
"""Compile a corrected float-I/O graph to a device-specific QNN ONNX context.

Requires qai-hub, onnx and an existing AI Hub client configuration. No credentials are logged.
Each invocation advances one stage without waiting for cloud compilation. Artifacts stay
outside app assets until numerical and device validation explicitly select them.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import qai_hub as hub
    import onnx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'app/src/main/assets/depth_anything_v2.onnx')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--device', default='Samsung Galaxy S21 (Family)')
    parser.add_argument('--qairt-version', default='2.45')
    parser.add_argument('--source-model-id', help='Reuse an already uploaded copy of this exact model')
    args = parser.parse_args()
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state['sha256'] != digest or state['device'] != args.device or state['qairt_version'] != args.qairt_version:
            raise SystemExit('State belongs to a different model/device/runtime; use a new state path')
    else:
        versions = [f.api_version for f in hub.get_frameworks() if f.name == 'QAIRT']
        if args.qairt_version not in versions:
            raise SystemExit(f'Requested QAIRT {args.qairt_version} unavailable; offered versions: {versions}')
        model = onnx.load(str(args.model))
        io_names = {v.name for v in [*model.graph.input, *model.graph.output]}
        removed = [v.name for v in model.graph.value_info if v.name in io_names]
        before_nodes = [n.SerializeToString() for n in model.graph.node]
        kept = [v for v in model.graph.value_info if v.name not in io_names]
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept)
        onnx.checker.check_model(model)
        assert before_nodes == [n.SerializeToString() for n in model.graph.node]
        prepared = args.state.parent / 'source.onnx'
        onnx.save(model, str(prepared))
        if args.source_model_id and removed:
            raise SystemExit('Cannot reuse a source upload before graph-I/O metadata normalization')
        state = dict(sha256=digest, prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
                     removed_io_value_info=removed, model=str(args.model.resolve()), device=args.device,
                     qairt_version=args.qairt_version)
        source = hub.get_model(args.source_model_id) if args.source_model_id else hub.upload_model(str(prepared))
        state['source_model_id'] = source.model_id
        args.state.write_text(json.dumps(state, indent=2))
    if 'compile_job_id' not in state:
        job = hub.submit_compile_job(
            model=hub.get_model(state['source_model_id']), device=hub.Device(args.device),
            name='depth_anything_v2_sm8350_context',
            options=f'--target_runtime precompiled_qnn_onnx --qairt_version {args.qairt_version}', retry=False)
        state['compile_job_id'] = job.job_id
        args.state.write_text(json.dumps(state, indent=2))
        print(json.dumps(state, indent=2))
        return 2
    job = hub.get_job(state['compile_job_id'])
    status = job.get_status()
    state['status'] = dict(code=status.code, message=status.message)
    args.state.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    if not status.finished:
        return 2
    if status.failure:
        if not state.get('failure_logs'):
            try:
                state['failure_logs'] = job.download_job_logs(str(args.state.parent / 'logs'))
                args.state.write_text(json.dumps(state, indent=2))
            except Exception as error:
                print(f'Could not download failure logs: {type(error).__name__}')
        return 1
    artifact = args.state.parent / 'depth_anything_v2_qnn.onnx'
    if not state.get('downloaded'):
        state['downloaded'] = job.download_target_model(str(artifact))
        args.state.write_text(json.dumps(state, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
