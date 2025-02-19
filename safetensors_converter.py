# Converts pickle files from pt/pth to safetensors format
# Version 2.0, no longer has the 4 GB file limit \o/

# Usage: python safetensor_converter.py <file/folder>

# Caution! Make sure you test the produced .safetensors files before deleting your original .pth files. Not all files can be converted successfully.

import os
import sys
import torch
from safetensors.torch import save_file
from colorama import Fore, Style, init
from typing import Any, Dict, Union, List, Tuple
from collections import OrderedDict
import importlib.metadata
from packaging import version


def check_safetensors_version() -> bool:
    """Checks if the installed safetensors version is at least 0.4.1 (required for handling files larger than 4 GB)"""
    safetensors_version = importlib.metadata.version('safetensors')
    if version.parse(safetensors_version) < version.parse("0.4.1"):
        return False
    return True


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> OrderedDict[str, Any]:
    """Self-explanatory, flattens nested, multi-level dicts to single-level ones"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, OrderedDict)):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)

def convert_to_float32(tensor: Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[str, Any]]) -> Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[str, Any]]:
    """Converts all tensor types to float32"""
    if isinstance(tensor, torch.Tensor):
        return tensor.float().contiguous()
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(convert_to_float32(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: convert_to_float32(v) for k, v in tensor.items()}
    return tensor

def get_state_dict(checkpoint: Union[torch.nn.Module, Dict[str, Any]]) -> Dict[str, Any]:
    """Gets a checkpoint's state dict"""
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint)
    raise ValueError("Unsupported checkpoint format")


def process_file(input_file: str, output_folder: str) -> int:
    """Main processing function, converts and saves the pickle. Returns 0, 1 or 2, depending on what happened - read the function's comments for info"""

    # Load the file and prepare the state dict
    checkpoint = torch.load(input_file, map_location="cpu", weights_only=True)
    state_dict = get_state_dict(checkpoint)
    flat_state_dict = flatten_dict(state_dict)
    flat_state_dict = convert_to_float32(flat_state_dict)

    try:
        save_file(flat_state_dict, output_folder)
        # I guess everything went fine right from the start? Return 1
        return 1
    except RuntimeError as e:
        # Of course it wouldn't go fine right from the start.
        if "non contiguous tensor" in str(e):
            # "safetensors (and many other serialization libraries) expect tensors to be stored in a contiguous block of memory. Sometimes,
            # PyTorch tensors can become non-contiguous due to certain operations (like slicing, transposing, etc., in some cases)."
            # Try to make the tensors contiguous and save again
            flat_state_dict = {
                k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in flat_state_dict.items()
            }
            try:
                save_file(flat_state_dict, output_folder)
                # Making the tensors contiguous worked, return 2 so we can inform the user about it
                return True
            except Exception as e:
                raise ValueError(
                    f"Failed to save the file in safetensors format: {e}"
                )
        elif "invalid load key" in str(e):
            # Corrupted file
            raise ValueError(
                "Invalid key found while trying to save. File is possibly corrupted and will be skipped."
            )
        else:
            raise  # Re-raise the original exception if it's not about non-contiguous tensors


def convert_pickle_to_safetensors(input_folder, file_extension):
    # Check output folder
    output_folder = os.path.join(input_folder, "converted_safetensors")
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Will save the converted files to {output_folder}")

    # Some informational lists and counters
    non_contiguous_tensors = []
    files_to_convert = [
        f for f in os.listdir(input_folder) if f.endswith(file_extension)
    ]
    num_files_to_convert = len(files_to_convert)
    logging.info(f"Found {num_files_to_convert} {file_extension} files to convert.")
    num_success = 0

    # Iterate + convert
    for num_current, filename in enumerate(files_to_convert, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(
            output_folder, f"{os.path.splitext(filename)[0]}.safetensors"
        )

        try:
            is_non_contiguous = process_file(input_path, output_path)
            num_success += 1
            log_message = f"{num_current} / {num_files_to_convert} - {filename}: Successfully converted to {os.path.basename(output_path)}"
            if is_non_contiguous:
                non_contiguous_tensors.append(os.path.basename(output_path))
                log_message += " but only after making tensors contiguous, needs to be checked afterwards."
            logging.info(log_message)
        except Exception as e:
            logging.error(
                f"{num_current} / {num_files_to_convert} - {filename}: {str(e)}"
            )

    logging.info(
        f"Successfully converted {num_success} files. {num_files_to_convert - num_success} files failed to convert."
    )
    if non_contiguous_tensors:
        logging.warning(
            f"The following files contained non-contiguous tensors that were made contiguous. Check that the .safetensors file works before deleting the original: {non_contiguous_tensors}"
        )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise ValueError(
            "Usage: python safetensor_converter.py <input_folder> [file_extension_to_convert (default: pth)]"
        )

    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        raise ValueError("Input folder does not exist")

    if len(sys.argv) == 3:
        file_extension = (
            f".{sys.argv[2]}" if not sys.argv[2].startswith(".") else sys.argv[2]
        )
    else:
        file_extension = ".pth"

    logging.info(f"Starting conversion of {file_extension} files in {input_folder} ...")
    convert_pickle_to_safetensors(input_folder, file_extension)
    logging.info("Conversion complete.")
