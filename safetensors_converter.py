"""Convert PyTorch model files (.pt/.pth) to .safetensors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from colorama import Fore, Style, init
from packaging import version
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file


warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

MIN_SAFETENSORS_VERSION = "0.4.1"
SUPPORTED_EXTENSIONS = (".pt", ".pth")


@dataclass(frozen=True)
class ConversionResult:
    status: str
    reason_code: str
    message: str
    input_file: str
    output_file: str
    validation_warnings: list[str]


@dataclass
class RuntimeOptions:
    allow_unsafe_load: bool
    ask_unsafe_once: bool
    cast_float32: bool
    verbose: bool
    validate: bool
    strict_validate: bool
    json_report: bool
    dry_run: bool


@dataclass(frozen=True)
class ValidationOutcome:
    success: bool
    errors: list[str]
    warnings: list[str]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Converts PyTorch model files (.pt/.pth) to .safetensors. "
            "Original files are never modified."
        )
    )
    parser.add_argument(
        "input_path",
        help="Single model file, or folder containing .pt/.pth files",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help=(
            "Output folder for converted files. "
            "Default: converted_safetensors inside the input folder"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra details while processing",
    )
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help=(
            "If weights_only=True loading fails, retry with weights_only=False "
            "without asking each time"
        ),
    )
    parser.add_argument(
        "--cast-float32",
        action="store_true",
        help=(
            "Cast floating tensors to float32 before saving (can improve compatibility with picky loaders/tools, but may increase file size and reduce precision)"
        ),
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip post-conversion validation checks",
    )
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help="Fail validation on any key/shape/dtype mismatch",
    )
    parser.add_argument(
        "--json-report",
        action="store_true",
        help="Write a detailed JSON report to the output folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what would be converted without loading model files or writing outputs"
        ),
    )

    args = parser.parse_args(argv[1:])
    if args.strict_validate and args.skip_validate:
        parser.error("--strict-validate cannot be combined with --skip-validate")
    return args


def resolve_paths(input_path_raw: str, output_dir_raw: str | None) -> tuple[Path, Path]:
    input_path = Path(input_path_raw).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if output_dir_raw:
        output_dir = Path(output_dir_raw).expanduser().resolve()
    elif input_path.is_file():
        output_dir = input_path.parent / "converted_safetensors"
    else:
        output_dir = input_path / "converted_safetensors"

    return input_path, output_dir


def check_safetensors_support() -> bool | None:
    installed = importlib.metadata.version("safetensors")
    supports_large_files = version.parse(installed) >= version.parse(
        MIN_SAFETENSORS_VERSION
    )

    if supports_large_files:
        print(
            Fore.GREEN + f"* safetensors {installed} supports files larger than 4 GB."
        )
        return True

    print(
        Fore.RED
        + Style.BRIGHT
        + "\n* Warning *\n"
        + Style.NORMAL
        + (
            f"Installed safetensors version ({installed}) can only handle models under 4 GB.\n"
            "Larger models will be skipped."
        )
    )
    user_input = (
        input(
            Fore.YELLOW
            + f"Continue with safetensors {installed} and skip >4 GB models? y/[n] :: "
        )
        .strip()
        .lower()
    )
    if user_input != "y":
        print(
            Fore.YELLOW
            + (
                f"\nUpdate safetensors to {MIN_SAFETENSORS_VERSION} or newer and run again.\n"
                "Exiting ...\n"
            )
        )
        return None

    print(Fore.YELLOW + "\n* Continuing with legacy size limitation active.\n")
    return False


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    return [
        p
        for p in sorted(input_path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def get_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, Mapping):
        return checkpoint.get("state_dict", checkpoint)
    return checkpoint


def extract_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}

    if isinstance(obj, torch.Tensor):
        key = prefix or "tensor"
        tensors[key] = obj
        return tensors

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            tensors.update(extract_tensors(value, next_prefix))
        return tensors

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            tensors.update(extract_tensors(value, next_prefix))
        return tensors

    return tensors


def prepare_tensors(
    tensors: dict[str, torch.Tensor], cast_float32: bool
) -> dict[str, torch.Tensor]:
    prepared: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        t = tensor.detach().cpu().contiguous()
        if cast_float32 and t.is_floating_point():
            t = t.float()
        prepared[name] = t
    return prepared


def tensor_bytes(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


def bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / (1024**3)


def load_checkpoint(
    input_file: Path,
    runtime: RuntimeOptions,
    unsafe_retry_enabled: bool,
) -> tuple[Any | None, bool, str | None]:
    try:
        return (
            torch.load(str(input_file), map_location="cpu", weights_only=True),
            unsafe_retry_enabled,
            None,
        )
    except TypeError:
        try:
            return (
                torch.load(str(input_file), map_location="cpu"),
                unsafe_retry_enabled,
                None,
            )
        except Exception as exc:
            return None, unsafe_retry_enabled, str(exc)
    except Exception as exc:
        initial_error = str(exc)

        should_try_unsafe = runtime.allow_unsafe_load
        if (
            not should_try_unsafe
            and runtime.ask_unsafe_once
            and not unsafe_retry_enabled
            and not runtime.dry_run
        ):
            print(
                Fore.YELLOW
                + Style.BRIGHT
                + "\n* Info: safe load failed for this file.\n"
                + Style.NORMAL
                + (
                    "You can retry with weights_only=False, which can execute code inside the model file.\n"
                    "Use this only for trusted model sources."
                )
            )
            user_input = (
                input(
                    Fore.YELLOW
                    + "Retry with weights_only=False for this run? y/[n] :: "
                )
                .strip()
                .lower()
            )
            should_try_unsafe = user_input == "y"
            unsafe_retry_enabled = should_try_unsafe

        if not should_try_unsafe:
            return None, unsafe_retry_enabled, initial_error

        try:
            return (
                torch.load(str(input_file), map_location="cpu", weights_only=False),
                unsafe_retry_enabled,
                None,
            )
        except TypeError:
            try:
                return (
                    torch.load(str(input_file), map_location="cpu"),
                    unsafe_retry_enabled,
                    None,
                )
            except Exception as unsafe_exc:
                return (
                    None,
                    unsafe_retry_enabled,
                    "Safe load failed and unsafe retry also failed:\n"
                    + f"weights_only=True error:\n{initial_error}\n"
                    + f"weights_only=False error:\n{unsafe_exc}",
                )
        except Exception as unsafe_exc:
            return (
                None,
                unsafe_retry_enabled,
                "Safe load failed and unsafe retry also failed:\n"
                + f"weights_only=True error:\n{initial_error}\n"
                + f"weights_only=False error:\n{unsafe_exc}",
            )


def validate_saved_output(
    prepared_tensors: dict[str, torch.Tensor],
    output_file: Path,
    strict: bool,
) -> ValidationOutcome:
    try:
        saved_tensors = load_safetensors_file(str(output_file), device="cpu")
    except Exception as exc:
        return ValidationOutcome(
            success=False,
            errors=[f"Could not load produced safetensors file: {exc}"],
            warnings=[],
        )

    src_keys = set(prepared_tensors.keys())
    dst_keys = set(saved_tensors.keys())

    missing_keys = sorted(src_keys - dst_keys)
    extra_keys = sorted(dst_keys - src_keys)

    errors: list[str] = []
    warnings: list[str] = []

    if missing_keys:
        errors.append(f"Missing {len(missing_keys)} tensor key(s) in output")
    if extra_keys:
        errors.append(f"Found {len(extra_keys)} extra tensor key(s) in output")

    common = sorted(src_keys & dst_keys)
    shape_mismatches = 0
    dtype_mismatches = 0

    for key in common:
        src = prepared_tensors[key]
        dst = saved_tensors[key]
        if tuple(src.shape) != tuple(dst.shape):
            shape_mismatches += 1
        if src.dtype != dst.dtype:
            dtype_mismatches += 1

    if shape_mismatches > 0:
        errors.append(f"{shape_mismatches} tensor shape mismatch(es)")

    if dtype_mismatches > 0:
        if strict:
            errors.append(f"{dtype_mismatches} tensor dtype mismatch(es)")
        else:
            warnings.append(f"{dtype_mismatches} tensor dtype mismatch(es)")

    return ValidationOutcome(
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def plan_dry_run(
    input_file: Path,
    output_dir: Path,
    supports_large_files: bool,
) -> ConversionResult:
    output_file = output_dir / f"{input_file.stem}.safetensors"

    notes: list[str] = []
    reason_code = "DRYRUN_READY"

    if output_file.exists():
        reason_code = "DRYRUN_OVERWRITE"
        notes.append("Output already exists and would be overwritten")

    if not supports_large_files:
        file_size = input_file.stat().st_size
        if file_size >= 4 * (1024**3):
            reason_code = "DRYRUN_SIZE_RISK"
            notes.append(
                "Input file itself is >= 4 GB, and current safetensors may fail depending on tensor payload size"
            )

    message = (
        "Dry run: conversion not executed"
        if not notes
        else "Dry run: " + "; ".join(notes)
    )

    return ConversionResult(
        status="DRYRUN",
        reason_code=reason_code,
        message=message,
        input_file=str(input_file),
        output_file=str(output_file),
        validation_warnings=[],
    )


def convert_file(
    input_file: Path,
    output_dir: Path,
    supports_large_files: bool,
    runtime: RuntimeOptions,
    unsafe_retry_enabled: bool,
) -> tuple[ConversionResult, bool]:
    if runtime.dry_run:
        return plan_dry_run(
            input_file, output_dir, supports_large_files
        ), unsafe_retry_enabled

    checkpoint, unsafe_retry_enabled, load_error = load_checkpoint(
        input_file, runtime, unsafe_retry_enabled
    )
    if load_error is not None:
        return (
            ConversionResult(
                status="FAILED",
                reason_code="FAIL_LOAD",
                message=f"Could not load checkpoint:\n{load_error}",
                input_file=str(input_file),
                output_file="",
                validation_warnings=[],
            ),
            unsafe_retry_enabled,
        )

    state_like = get_state_dict(checkpoint)
    tensors = extract_tensors(state_like)
    if not tensors:
        return (
            ConversionResult(
                status="FAILED",
                reason_code="FAIL_NO_TENSORS",
                message="No tensors found in loaded object. Unsupported checkpoint structure.",
                input_file=str(input_file),
                output_file="",
                validation_warnings=[],
            ),
            unsafe_retry_enabled,
        )

    prepared = prepare_tensors(tensors, cast_float32=runtime.cast_float32)

    if not supports_large_files:
        size = tensor_bytes(prepared)
        if size >= 4 * (1024**3):
            return (
                ConversionResult(
                    status="SKIPPED",
                    reason_code="SKIP_SIZE_LIMIT",
                    message=(
                        "Tensor payload exceeds 4 GB "
                        f"({bytes_to_gb(size):.2f} GB). Current safetensors version cannot save it."
                    ),
                    input_file=str(input_file),
                    output_file="",
                    validation_warnings=[],
                ),
                unsafe_retry_enabled,
            )

    output_file = output_dir / f"{input_file.stem}.safetensors"

    try:
        save_file(prepared, str(output_file))
    except Exception as exc:
        err = str(exc)
        if "invalid load key" in err.lower():
            return (
                ConversionResult(
                    status="FAILED",
                    reason_code="FAIL_INVALID_FORMAT",
                    message=f"Invalid/corrupted input or unsupported format:\n{err}",
                    input_file=str(input_file),
                    output_file="",
                    validation_warnings=[],
                ),
                unsafe_retry_enabled,
            )
        if "non contiguous" in err.lower():
            return (
                ConversionResult(
                    status="FAILED",
                    reason_code="FAIL_NONCONTIG",
                    message=f"Failed after preparing contiguous tensors:\n{err}",
                    input_file=str(input_file),
                    output_file="",
                    validation_warnings=[],
                ),
                unsafe_retry_enabled,
            )
        return (
            ConversionResult(
                status="FAILED",
                reason_code="FAIL_SAVE",
                message=f"Save failed for an unexpected reason:\n{err}",
                input_file=str(input_file),
                output_file="",
                validation_warnings=[],
            ),
            unsafe_retry_enabled,
        )

    if runtime.validate:
        validation = validate_saved_output(
            prepared_tensors=prepared,
            output_file=output_file,
            strict=runtime.strict_validate,
        )
        if not validation.success:
            return (
                ConversionResult(
                    status="FAILED",
                    reason_code="FAIL_VALIDATE",
                    message="Validation failed: " + "; ".join(validation.errors),
                    input_file=str(input_file),
                    output_file=str(output_file),
                    validation_warnings=validation.warnings,
                ),
                unsafe_retry_enabled,
            )

        if validation.warnings:
            return (
                ConversionResult(
                    status="OK",
                    reason_code="OK_VALIDATED_WARN",
                    message="Converted and validated with warnings",
                    input_file=str(input_file),
                    output_file=str(output_file),
                    validation_warnings=validation.warnings,
                ),
                unsafe_retry_enabled,
            )

        return (
            ConversionResult(
                status="OK",
                reason_code="OK_VALIDATED",
                message="Converted and validated",
                input_file=str(input_file),
                output_file=str(output_file),
                validation_warnings=[],
            ),
            unsafe_retry_enabled,
        )

    return (
        ConversionResult(
            status="OK",
            reason_code="OK_NO_VALIDATE",
            message="Converted (validation skipped)",
            input_file=str(input_file),
            output_file=str(output_file),
            validation_warnings=[],
        ),
        unsafe_retry_enabled,
    )


def print_file_status(
    idx: int,
    total: int,
    model_file: Path,
    result: ConversionResult,
    verbose: bool,
) -> None:
    if result.status == "OK":
        color = Fore.GREEN
    elif result.status == "SKIPPED":
        color = Fore.YELLOW
    elif result.status == "DRYRUN":
        color = Fore.CYAN
    else:
        color = Fore.RED

    base = (
        f"[{str(idx).zfill(3)}/{str(total).zfill(3)}] "
        f"{model_file.name} -> {result.status} [{result.reason_code}]"
    )
    print(color + base)

    if verbose:
        print(Style.NORMAL + f"    {result.message}")
        for warning in result.validation_warnings:
            print(Fore.YELLOW + f"    validation warning: {warning}")


def print_final_report(
    results: list[ConversionResult],
    elapsed_seconds: float,
    runtime: RuntimeOptions,
    output_dir: Path,
    json_path: Path | None,
) -> None:
    reason_legend = {
        "OK_VALIDATED": "Converted and validation passed",
        "OK_VALIDATED_WARN": "Converted and validated with warnings",
        "OK_NO_VALIDATE": "Converted without validation",
        "SKIP_SIZE_LIMIT": "Skipped due to legacy 4 GB limit",
        "FAIL_LOAD": "Could not load checkpoint",
        "FAIL_NO_TENSORS": "No tensors found in checkpoint",
        "FAIL_INVALID_FORMAT": "Invalid/corrupted input format",
        "FAIL_NONCONTIG": "Could not save after contiguous prep",
        "FAIL_SAVE": "Save failed for another reason",
        "FAIL_VALIDATE": "Post-save validation failed",
        "DRYRUN_READY": "Dry run: ready to convert",
        "DRYRUN_OVERWRITE": "Dry run: output exists and would be overwritten",
        "DRYRUN_SIZE_RISK": "Dry run: potential size limitation risk",
    }
    failure_actions = {
        "FAIL_LOAD": "Try --allow-unsafe-load only for trusted files; if still failing, verify file integrity/source.",
        "FAIL_NO_TENSORS": "This file is likely not a plain tensor checkpoint; inspect its structure before converting.",
        "FAIL_INVALID_FORMAT": "Check that the file is a valid .pt/.pth checkpoint and re-download if corruption is suspected.",
        "FAIL_NONCONTIG": "Re-save the original checkpoint from PyTorch if possible, then retry conversion.",
        "FAIL_SAVE": "Re-run with --verbose for detail and verify disk permissions/free space.",
        "FAIL_VALIDATE": "Run again with --verbose and inspect key/shape/dtype mismatches before using the output.",
    }

    status_counts = Counter(r.status for r in results)
    reason_counts = Counter(r.reason_code for r in results)

    print(Fore.CYAN + Style.BRIGHT + "\n=| Run Report |=\n")

    print(Fore.CYAN + Style.BRIGHT + "Summary")
    print(Fore.CYAN + f"* Total files considered: {len(results)}")
    print(Fore.GREEN + f"* OK: {status_counts.get('OK', 0)}")
    print(Fore.YELLOW + f"* SKIPPED: {status_counts.get('SKIPPED', 0)}")
    print(Fore.RED + f"* FAILED: {status_counts.get('FAILED', 0)}")
    print(Fore.CYAN + f"* DRYRUN: {status_counts.get('DRYRUN', 0)}")
    print(Fore.CYAN + f"* Validation: {'ON' if runtime.validate else 'OFF'}")
    print(
        Fore.CYAN
        + f"* Validation strict mode: {'ON' if runtime.strict_validate else 'OFF'}"
    )
    print(Fore.CYAN + f"* Elapsed: {elapsed_seconds:.2f}s")

    print(Fore.CYAN + Style.BRIGHT + "\nReason Code Breakdown")
    for reason, count in sorted(reason_counts.items(), key=lambda item: item[0]):
        print(Fore.CYAN + f"* {reason}: {count}")

    print(Fore.CYAN + Style.BRIGHT + "\nReason Code Legend")
    for reason in sorted(reason_counts.keys()):
        description = reason_legend.get(reason, "No legend entry available")
        print(Fore.CYAN + f"* {reason}: {description}")

    failures = [r for r in results if r.status == "FAILED"]
    skips = [r for r in results if r.status == "SKIPPED"]
    warns = [r for r in results if r.validation_warnings]

    if failures:
        print(Fore.RED + Style.BRIGHT + "\nFailures")
        for item in failures:
            print(Fore.RED + f"* {Path(item.input_file).name}: {item.reason_code}")
            action = failure_actions.get(
                item.reason_code,
                "Check verbose output and source checkpoint integrity, then retry.",
            )
            print(Fore.YELLOW + f"  suggested action: {action}")
            if runtime.verbose:
                print(Fore.RED + f"  {item.message}")

    if skips:
        print(Fore.YELLOW + Style.BRIGHT + "\nSkipped")
        for item in skips:
            print(Fore.YELLOW + f"* {Path(item.input_file).name}: {item.reason_code}")

    if warns:
        print(Fore.YELLOW + Style.BRIGHT + "\nValidation Warnings")
        for item in warns:
            print(Fore.YELLOW + f"* {Path(item.input_file).name}:")
            for warning in item.validation_warnings:
                print(Fore.YELLOW + f"  - {warning}")

    print(Fore.CYAN + Style.BRIGHT + "\nOutput")
    print(Fore.CYAN + f"* Output folder: {output_dir}")
    if json_path is not None:
        print(Fore.CYAN + f"* JSON report: {json_path}")
    else:
        print(Fore.CYAN + "* JSON report: disabled")


def write_results_json(output_dir: Path, details: list[dict[str, Any]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"_results_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    return path


def main(argv: list[str]) -> int:
    init(autoreset=True)

    print(
        Fore.CYAN
        + "\n "
        + "-" * 39
        + "\n--| "
        + Style.BRIGHT
        + "SafeTensors Converter Script"
        + Style.NORMAL
        + " |--\n"
        + " "
        + "-" * 39
        + "\n"
    )

    args = parse_args(argv)

    try:
        input_path, output_dir = resolve_paths(args.input_path, args.output_dir)
    except Exception as exc:
        print(Fore.RED + Style.BRIGHT + "Error! " + Style.NORMAL + str(exc))
        return 1

    if not args.dry_run:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Error! "
                + Style.NORMAL
                + f"Could not create output directory {output_dir}:\n{exc}"
            )
            return 1

    if args.verbose:
        print(Fore.CYAN + "* Checking installed safetensors version ...")

    supports_large_files = check_safetensors_support()
    if supports_large_files is None:
        return 1

    files = collect_input_files(input_path)
    if not files:
        print(Fore.YELLOW + "No .pt/.pth files found to process.")
        return 0

    runtime = RuntimeOptions(
        allow_unsafe_load=args.allow_unsafe_load,
        ask_unsafe_once=True,
        cast_float32=args.cast_float32,
        verbose=args.verbose,
        validate=not args.skip_validate,
        strict_validate=args.strict_validate,
        json_report=args.json_report,
        dry_run=args.dry_run,
    )

    if runtime.verbose:
        mode = "file" if input_path.is_file() else "folder"
        print(Fore.CYAN + "* Run configuration:")
        print(Fore.CYAN + f"  + mode:               {mode}")
        print(Fore.CYAN + f"  + input:              {input_path}")
        print(Fore.CYAN + f"  + output:             {output_dir}")
        print(Fore.CYAN + f"  + files:              {len(files)}")
        print(Fore.CYAN + f"  + cast float32:       {runtime.cast_float32}")
        print(Fore.CYAN + f"  + validate:           {runtime.validate}")
        print(Fore.CYAN + f"  + strict validate:    {runtime.strict_validate}")
        print(Fore.CYAN + f"  + dry run:            {runtime.dry_run}")
        print(Fore.CYAN + f"  + json report:        {runtime.json_report}")

    print(
        Fore.CYAN
        + (
            f"\n* Planning {len(files)} model file(s) ..."
            if runtime.dry_run
            else f"\n* Processing {len(files)} model file(s) ..."
        )
    )

    started = time.perf_counter()
    unsafe_retry_enabled = runtime.allow_unsafe_load

    all_results: list[ConversionResult] = []

    for idx, model_file in enumerate(files, start=1):
        result, unsafe_retry_enabled = convert_file(
            model_file,
            output_dir,
            supports_large_files,
            runtime,
            unsafe_retry_enabled,
        )
        all_results.append(result)
        print_file_status(idx, len(files), model_file, result, runtime.verbose)

    elapsed = time.perf_counter() - started

    json_path: Path | None = None
    if runtime.json_report:
        json_rows: list[dict[str, Any]] = []
        for r in all_results:
            json_rows.append({
                "input_file": r.input_file,
                "output_file": r.output_file,
                "status": r.status,
                "reason_code": r.reason_code,
                "message": r.message,
                "validation_warnings": r.validation_warnings,
            })
        if not runtime.dry_run:
            json_path = write_results_json(output_dir, json_rows)
        else:
            # For dry runs, use a sibling report file without creating conversion output folders.
            fallback_dir = output_dir if output_dir.exists() else input_path.parent
            fallback_dir.mkdir(parents=True, exist_ok=True)
            json_path = write_results_json(fallback_dir, json_rows)

    print_final_report(
        results=all_results,
        elapsed_seconds=elapsed,
        runtime=runtime,
        output_dir=output_dir,
        json_path=json_path,
    )

    if runtime.dry_run:
        print(
            Fore.CYAN + "\nDry run finished. No model files were loaded or converted.\n"
        )
        return 0

    if any(r.status == "FAILED" for r in all_results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
