# Script that converts PyTorch model files (.pt, .pth) to the safer .safetensors format.
# Version 2.0, no longer has the 4 GB file limit \o/

# Usage: python safetensors_converter.py <input_file/folder> [output_folder]
# Caution! Make sure you test the produced .safetensors files before deleting your original files. Not all files can be converted successfully.

import os
import sys
import torch
from safetensors.torch import save_file
from colorama import Fore, Style, init
from typing import Any, Dict, Union, List, Tuple
from collections import OrderedDict
import importlib.metadata
from packaging import version


def check_safetensors_version() -> Tuple[bool, str]:
    """Checks if the installed safetensors version is at least 0.4.1 (required for handling files larger than 4 GB)"""
    safetensors_version = importlib.metadata.version('safetensors')
    if version.parse(safetensors_version) < version.parse("0.4.1"):
        return (False, safetensors_version)
    return (True, safetensors_version)


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


def convert_file(input_file: str, output_file: str) -> bool:
    """Conversion function, receives a model file, converts it to .safetensors and saves it"""
    # Load the file and prepare the state dict
    try:
        state_dict = get_state_dict(torch.load(input_file, map_location="cpu", weights_only=True))
    except Exception as e:
        raise ValueError(f"Could not load the input file and retrieve its state dict: {e}")
    processed_state_dict = convert_to_float32(flatten_dict(state_dict))

    try:
        save_file(processed_state_dict, output_file)
        # It just worked I guess
        return True
    except RuntimeError as e:
        # Of course it wouldn't work
        if "non contiguous tensor" in str(e):
            # Try to make the tensors contiguous and save again
            processed_state_dict = {
                k: v.contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in processed_state_dict.items()
            }
            try:
                save_file(processed_state_dict, output_file)
                # Making the tensors contiguous worked
                return False
            except Exception as e:
                raise ValueError(f"Failed to save the file in safetensors format: {e}")
        elif "invalid load key" in str(e):
            # Invalid/corrupted file
            raise ValueError("Invalid key found while trying to save as safetensors. The model file is either in invalid format or corrupted, and will be skipped.")
        else:
            raise  # Re-raise the original exception if it's not about non-contiguous tensors


def main_processor(input_folder: str, input_file: str | None, output_folder: str):
    """Main processing function, handles the input(s) and calls the conversion function as necessary"""




if __name__ == "__main__":
    init(autoreset=True)
    print(Fore.CYAN + Style.BRIGHT + "\n**| SafeTensors Converter Script - v2 |**\n")

    # Show help message and exit if no args or if --help/-h is passed
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("--help", "-h")):
        print(Fore.CYAN + "\nPython script to convert PyTorch model files (.pt, .pth) to the safer .safetensors format.\n")
        print(Fore.CYAN + Style.BRIGHT + "Usage:")
        print(Fore.CYAN + "python " + Style.BRIGHT + "safetensors_converter.py " + Fore.YELLOW + "<input_file/folder>" + Style.NORMAL + " [output_folder]\n")
        print(Fore.CYAN + Style.BRIGHT + "Arguments:")
        print(Fore.CYAN + Style.BRIGHT + "* input_file/folder:" + Fore.YELLOW + Style.NORMAL + "\n\tRequired.\n\tEither a single file or a folder containing .pt or .pth files to convert.")
        print(Fore.CYAN + Style.BRIGHT + "* output_folder:" + Fore.YELLOW + Style.NORMAL + "\n\tOptional.\n\tThe folder to save the converted files to.\n\tDefault: Subfolder 'converted_safetensors', created in the input folder/the input file's folder.\n")
        print(Fore.MAGENTA + Style.BRIGHT + "Important note:\n" + Style.NORMAL + "Not all models can be converted to functioning .safetensors versions.\nThe script won't delete or otherwise modify the original files.\n" + Style.BRIGHT + "Make sure you test the produced .safetensors files before deleting the original files!\n")
        sys.exit(0)

    # Check safetensors version and ask user what to do
    ok_sft_version, sft_version_str = check_safetensors_version()
    if not ok_sft_version:
        print(Fore.RED + Style.BRIGHT + f"The currently installed safetensors library version is {sft_version_str}, which is older than 0.4.1 and can only handle files smaller than 4 GB.")
        print(Fore.RED + "You may proceed with the current version if the files you want to convert are smaller than 4 GB, otherwise please exit the script and update your safetensors library (e.g. with: pip install \"safetensors>=0.4.1\").\n")
        user_input = input(Fore.RED + f"Proceed with current version {sft_version_str} and convert only files smaller than 4 GB? y/[n] :: ")
        if user_input.lower().strip() != "y":
            sys.exit(1)

    # Parse input argument, must be an existing file or folder
    input_path = os.path.abspath(sys.argv[1])
    if os.path.isfile(input_path):
        # Single file
        input_folder = os.path.dirname(input_path)
        input_file = input_path
    elif os.path.isdir(input_path):
        # Folder
        input_folder = input_path
        input_file = None
    else:
        print(Fore.RED + Style.BRIGHT + "\nError! " + Style.NORMAL + f"The input argument ({input_path}) is not an existing file or folder. Exiting ...\n")
        sys.exit(1)

    # Check output path argument and create output folder
    if len(sys.argv) == 2:
        output_path = os.path.join(input_folder, "converted_safetensors")
    else:
        output_path = os.path.abspath(sys.argv[2])

    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + "\nError! " + Style.NORMAL + f"Couldn't create the output folder ({output_path}):\n{e}\nExiting ...\n")

    # Start processing
    probably_stats_idk_yet = process_input(input_folder, input_file, output_path)