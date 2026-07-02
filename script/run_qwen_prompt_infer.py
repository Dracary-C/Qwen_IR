#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_config import DEFAULT_CONFIG_PATH, config_get, load_config
from module.runtime import build
from module.runtime.paths import require_exists

DEFAULT_QWEN_CONFIG = ROOT / 'config' / 'tpgdiff_fewshot_qwen3vl.yml'
DEFAULT_QWEN_EXPORT = ROOT / 'module' / 'qwen' / 'export_tpgdiff_qwen_structured_dataset.py'
DEFAULT_QWEN_TMP_ROOT = Path('/data/chenzt/Dataset/Qwen3VL/tmp_test/myfusion_qwen_prompt_single')


def _is_set(value: Any) -> bool:
    return value not in (None, '', '~', 'null', 'None')


def _path_value(value: Any) -> Path | None:
    if not _is_set(value):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _safe_stem(path: Path) -> str:
    stem = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in path.stem).strip('_')
    return stem or 'input'


def _single_relative_image(image: Path) -> Path:
    digest_src = f'{image.resolve()}:{image.stat().st_mtime_ns}:{image.stat().st_size}'
    digest = hashlib.sha1(digest_src.encode('utf-8')).hexdigest()[:10]
    return Path('Single') / 'LQ' / f'{_safe_stem(image)}_{digest}{image.suffix.lower()}'


def _load_qwen_prompt_json(path: Path) -> list[float]:
    payload = json.loads(path.expanduser().read_text(encoding='utf-8'))
    vector = payload.get('prior_vector_21') if isinstance(payload, dict) else None
    if vector is None and isinstance(payload, dict):
        vector = payload.get('qwen_prompt_prior', payload.get('structured_prior'))
    if not isinstance(vector, list) or len(vector) != 21:
        raise SystemExit(f'qwen_prompt JSON must contain prior_vector_21 with 21 values: {path}')
    return [float(value) for value in vector]


def _run_qwen_online(cfg: dict[str, Any], image: Path, output_dir: Path) -> Path:
    qwen_opt = config_get(cfg, 'qwen_prompt', {}) or {}
    if not bool(qwen_opt.get('online', False)):
        raise SystemExit('No qwen prompt JSON was configured. Set paths.qwen_prompt_json or enable qwen_prompt.online.')

    output_root = Path(qwen_opt.get('output_root') or DEFAULT_QWEN_TMP_ROOT).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    relative_image = _single_relative_image(image)
    json_path = output_root / relative_image.with_suffix('.json')
    if json_path.exists() and not bool(qwen_opt.get('overwrite', False)):
        return json_path

    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'myfusion_qwen_prompt_single',
        'targets': [
            {
                'split': 'Single',
                'dataset': 'Single',
                'image': str(image.resolve()),
                'relative_image': str(relative_image),
                'expected_main_degradation': '',
            }
        ],
    }
    manifest_path = output_dir / f'{_safe_stem(image)}_qwen_target_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    python_bin = str(config_get(cfg, 'commands.python', sys.executable))
    export_script = _path_value(qwen_opt.get('export_script') or DEFAULT_QWEN_EXPORT)
    qwen_config = _path_value(qwen_opt.get('config') or DEFAULT_QWEN_CONFIG)
    cmd = [
        python_bin,
        str(export_script),
        '--config',
        str(qwen_config),
        '--data-root',
        str(image.parent.resolve()),
        '--output-root',
        str(output_root),
        '--target-manifest',
        str(manifest_path),
    ]
    if _is_set(qwen_opt.get('gpu')):
        cmd += ['--gpu', str(qwen_opt.get('gpu'))]
    if bool(qwen_opt.get('overwrite', False)):
        cmd.append('--overwrite')

    env = os.environ.copy()
    if _is_set(qwen_opt.get('gpu')):
        env['CUDA_VISIBLE_DEVICES'] = str(qwen_opt.get('gpu'))
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    log_path = output_dir / f'{_safe_stem(image)}_qwen_online.log'
    with log_path.open('a', encoding='utf-8') as log_handle:
        log_handle.write('\n# qwen_prompt single-image online run\n')
        log_handle.write(' '.join(cmd) + '\n')
        log_handle.flush()
        subprocess.run(cmd, check=True, env=env, stdout=log_handle, stderr=subprocess.STDOUT)

    if not json_path.exists():
        raise SystemExit(f'Qwen online generation did not create expected JSON: {json_path}')
    return json_path


def _resolve_checkpoint(cfg: dict[str, Any], override: Path | None) -> Path:
    value = override or _path_value(config_get(cfg, 'run.checkpoint'))
    if value is None:
        value = _path_value(config_get(cfg, 'paths.assess_tpgd_checkpoint'))
    if value is None:
        value = _path_value(config_get(cfg, 'path.checkpoint_load'))
    if value is None:
        raise SystemExit('Set run.checkpoint, paths.assess_tpgd_checkpoint, path.checkpoint_load, or pass --checkpoint.')
    return require_exists(value, 'Assess-TPGD/QwenPrompt checkpoint')


def _resize_image(image: Image.Image, side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert('RGB')
    if side and side > 0:
        image = image.resize((side, side), Image.Resampling.BICUBIC)
    return image


def _write_runtime_options(cfg: dict[str, Any], base_options: Path, run_dir: Path) -> Path:
    base = yaml.safe_load(base_options.read_text(encoding='utf-8')) or {}
    if not isinstance(base, dict):
        raise SystemExit(f'TPGD options must contain a YAML mapping: {base_options}')
    opt = deepcopy(base)

    prior_switch = config_get(cfg, 'prior_switch', {}) or {}
    opt['prior_switch'] = {
        'use_deg_prior': True,
        'use_content_prior': bool(prior_switch.get('use_content_prior', False)),
        'use_struct_prior': bool(prior_switch.get('use_struct_prior', True)),
    }

    network_g = deepcopy(config_get(cfg, 'network_G', {}) or {})
    if network_g:
        setting = network_g.setdefault('setting', {})
        setting.setdefault('use_degra_context', True)
        setting['use_image_context'] = bool(prior_switch.get('use_content_prior', False))
        setting['use_struct_context'] = bool(prior_switch.get('use_struct_prior', True))
        opt['network_G'] = network_g

    structure_prior = deepcopy(config_get(cfg, 'structure_prior', {}) or {})
    if structure_prior:
        opt['structure_prior'] = structure_prior

    sde = deepcopy(config_get(cfg, 'sde', {}) or {})
    if sde:
        opt['sde'] = {**(opt.get('sde', {}) or {}), **sde}

    degradation = deepcopy(config_get(cfg, 'degradation', {}) or {})
    if degradation:
        opt['degradation'] = {**(opt.get('degradation', {}) or {}), **degradation}

    opt.setdefault('path', {})
    opt['path']['strict_load'] = bool(config_get(cfg, 'path.strict_load', False))
    runtime_options = run_dir / 'runtime_qwen_prompt_options.yml'
    runtime_options.write_text(yaml.safe_dump(opt, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return runtime_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Single-pass TPGD + Qwen structured prior inference.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--input', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--qwen-prompt-json', type=Path, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--sampling-mode', choices=['posterior', 'sde'], default=None)
    parser.add_argument("--boundary-pad", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cuda_visible = config_get(cfg, 'runtime.cuda_visible_devices', None)
    if _is_set(cuda_visible) and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)
    tokenizers_parallelism = config_get(cfg, 'runtime.tokenizers_parallelism', False)
    os.environ['TOKENIZERS_PARALLELISM'] = str(tokenizers_parallelism).lower()

    input_path = require_exists(args.input or Path(str(config_get(cfg, 'paths.input'))), 'input image')
    output_dir = args.output_dir or _path_value(config_get(cfg, 'paths.qwen_prompt_output_dir'))
    if output_dir is None:
        output_dir = _path_value(config_get(cfg, 'paths.output_dir')) or (ROOT / 'outputs' / 'qwen_prompt_run')
    run_dir = output_dir.expanduser() / datetime.now().strftime('%Y%m%d_%H%M%S') / _safe_stem(input_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    qwen_json = args.qwen_prompt_json or _path_value(config_get(cfg, 'paths.qwen_prompt_json'))
    if qwen_json is None:
        qwen_json = _path_value(config_get(cfg, 'paths.structured_prior_json'))
    if qwen_json is None:
        qwen_json = _run_qwen_online(cfg, input_path, run_dir)
    qwen_json = require_exists(qwen_json, 'qwen_prompt JSON')
    qwen_prior = _load_qwen_prompt_json(qwen_json)

    checkpoint = _resolve_checkpoint(cfg, args.checkpoint)
    base_tpgd_options = require_exists(Path(str(config_get(cfg, 'paths.tpgd_options'))), 'TPGD runtime options')
    tpgd_options = _write_runtime_options(cfg, base_tpgd_options, run_dir)
    tpgd_checkpoint = require_exists(Path(str(config_get(cfg, 'paths.tpgd_checkpoint'))), 'TPGD restoration checkpoint')
    tpgd_prior = require_exists(Path(str(config_get(cfg, 'paths.tpgd_prior'))), 'TPGD prior checkpoint')

    device = args.device or str(config_get(cfg, 'fusion.device', 'cuda'))
    sampling_mode = args.sampling_mode or str(config_get(cfg, 'fusion.sampling_mode', 'posterior'))
    resize = int(args.resize if args.resize is not None else config_get(cfg, 'fusion.resize', 256))
    boundary_pad = int(args.boundary_pad if args.boundary_pad is not None else config_get(cfg, "fusion.boundary_pad", config_get(cfg, "runtime.boundary_pad", 32)))

    image = _resize_image(Image.open(input_path), resize)
    input_copy = run_dir / f'{_safe_stem(input_path)}_input.png'
    output_path = run_dir / f'{_safe_stem(input_path)}_qwen_prompt.png'
    metadata_path = run_dir / f'{_safe_stem(input_path)}_qwen_prompt.json'
    image.save(input_copy)

    runtime = build(
        'tpgdiff-runtime',
        options_path=tpgd_options,
        restoration_checkpoint=tpgd_checkpoint,
        tpgd_checkpoint=tpgd_prior,
        device=device,
        sampling_mode=sampling_mode,
        assessment_checkpoint=checkpoint,
        adapter_hidden_dim=int(config_get(cfg, 'test.adapter_hidden_dim', config_get(cfg, 'train.adapter_hidden_dim', 1024))),
        adapter_pool=str(config_get(cfg, 'test.adapter_pool', config_get(cfg, 'train.adapter_pool', 'mean'))),
        adapter_dropout=float(config_get(cfg, 'test.adapter_dropout', config_get(cfg, 'train.adapter_dropout', 0.0))),
        strict_load=bool(config_get(cfg, 'path.strict_load', False)),
        image_range=str(config_get(cfg, 'fusion.image_range', 'zero_one')),
        boundary_pad=boundary_pad,
    )
    runtime.load()
    output = runtime.restore(image, qwen_prompt_prior=qwen_prior)
    output.save(output_path)

    metadata = {
        'mode': 'qwen_prompt',
        'config': str(args.config.expanduser().resolve()),
        'input': str(input_path),
        'input_copy': str(input_copy),
        'output': str(output_path),
        'qwen_prompt_json': str(qwen_json),
        'checkpoint': str(checkpoint),
        'base_tpgd_options': str(base_tpgd_options),
        'runtime_tpgd_options': str(tpgd_options),
        'tpgd_checkpoint': str(tpgd_checkpoint),
        'tpgd_prior': str(tpgd_prior),
        'device': device,
        'sampling_mode': sampling_mode,
        'resize': resize,
        'image_range': str(config_get(cfg, 'fusion.image_range', 'zero_one')),
        "boundary_pad": boundary_pad,
        'prior_vector_order': [
            'noise_severity', 'blur_severity', 'haze_severity', 'rain_severity', 'low_light_severity',
            'noise_prob', 'blur_prob', 'haze_prob', 'rain_prob', 'low_light_prob',
            'global', 'local_region', 'object_specific', 'continuous', 'discrete',
            'directional', 'depth_dependent', 'shadow_dependent', 'texture_dependent', 'uncertain',
            'probability_margin',
        ],
        'prior_vector_21': qwen_prior,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'input={input_copy}')
    print(f'output={output_path}')
    print(f'metadata={metadata_path}')
    print(f'qwen_prompt_json={qwen_json}')


if __name__ == '__main__':
    main()
