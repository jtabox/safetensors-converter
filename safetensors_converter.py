# Converts pickles to safetensors.
# Usage: python safetensor_converter.py <input_folder> [file_extension_to_convert (default: pth)]

# Make sure you test the produced .safetensors files before deleting your original .pth files.

import os
import sys
import torch
import logging
from safetensors.torch import save_file
from collections import OrderedDict

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s:: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


# Well yeah, it flattens dicts :D
def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, OrderedDict)):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)


# Converts tensors to float32 (required for safetensors apparently)
def convert_to_float32(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.float().contiguous()
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(convert_to_float32(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: convert_to_float32(v) for k, v in tensor.items()}
    return tensor


# Gets the checkpoint's state dict
def get_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint)
    raise ValueError("Unsupported checkpoint format")


# Does the main processing and saving for each file, returns various values (see comments)
def process_file(input_path, output_path):
    # Load, get state dict, flatten and convert to float32
    checkpoint = torch.load(input_path, map_location="cpu")
    state_dict = get_state_dict(checkpoint)
    flat_state_dict = flatten_dict(state_dict)
    flat_state_dict = convert_to_float32(flat_state_dict)

    # Check file size
    total_size = sum(
        t.numel() * t.element_size()
        for t in flat_state_dict.values()
        if isinstance(t, torch.Tensor)
    )
    if total_size > 4 * (1024**3):  # 4 GB limit
        raise ValueError(
            f"File size ({total_size / (1024 ** 3):.2f} GB) exceeds safetensors' 4 GB limit."
        )

    try:
        save_file(flat_state_dict, output_path)
        # The file was saved without incidents, returns False
        return False
    except RuntimeError as e:
        # Of course there'd be incidents
        if "non contiguous tensor" in str(e):
            # Try to convert the tensors to contiguous and save again
            flat_state_dict = {
                k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in flat_state_dict.items()
            }
            try:
                save_file(flat_state_dict, output_path)
                # The file was saved after making tensors contiguous, returns True so the user can check it
                return True
            except Exception as e:
                raise ValueError(
                    f"Failed to save even after making tensors contiguous: {str(e)}"
                )
        elif "invalid load key" in str(e):
            # Tbh I'm not sure it's important to inform the user about this, but whatever, maybe they think it's something on their end
            raise ValueError(
                "Invalid dict key found while loading. File is possibly corrupted and will be skipped."
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
