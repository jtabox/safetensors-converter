# Script that converts PyTorch model files (.pt, .pth) to the safer .safetensors format.
# Version 2.0, no longer has the 4 GB file limit \o/

# Usage: python safetensors_converter.py <input file/folder> [output folder] [--verbose]
# Caution! Make sure you test the produced .safetensors files before deleting your original files! Not all files can be converted successfully.

import os
import sys
import torch
import json
from datetime import datetime
from safetensors.torch import save_file
from colorama import Fore, Style, init
from typing import Any, Dict, Union, List, Tuple
from collections import OrderedDict
import importlib.metadata
from packaging import version
import logging
import warnings


# Need torch to stfu because I can't take it anymore
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)


def parse_args(args: list) -> Tuple[str, str | None, str, bool] | None:
    """Parses the command line arguments and returns the input folder, input file, output folder and verbose mode"""
    # At least 1 arg given, it may only be an input file or folder
    input_path = os.path.abspath(args[1])
    if os.path.isfile(input_path):
        # Single file
        input_folder = os.path.dirname(input_path)
        input_file = input_path
    elif os.path.isdir(input_path):
        # Folder
        input_folder = input_path
        input_file = None
    else:
        print(
            Fore.RED
            + Style.BRIGHT
            + "Error! "
            + Style.NORMAL
            + f"The input argument ({input_path}) is not an existing file or folder.\nCorrect it and re-run. Exiting ...\n"
        )
        return None

    # Default output path and verbose mode. If they're not changed by the ifs below, they'll be used as is
    output_path = os.path.join(input_folder, "converted_safetensors")
    verbose_mode = False

    # Check for 2nd + 3rd args, only valid combos are:
    # 2nd -> output folder and 3rd -> verbose flag, or
    # 2nd -> verbose flag. Anything else after the flag is ignored.
    if len(args) >= 3:
        # 2nd exists
        if args[2].strip().lower() == "--verbose":
            # is the verbose flag
            verbose_mode = True
        elif len(args) >= 4:
            # 3rd exists, assume 2nd is the output folder
            output_path = os.path.abspath(args[2])
            if args[3].strip().lower() == "--verbose":
                # 3rd is the verbose flag
                verbose_mode = True
        else:
            # at least 2 args were passed, the 2nd isn't verbose flag, assume it's the output folder name
            output_path = os.path.abspath(args[2])

    return input_folder, input_file, output_path, verbose_mode


def show_help_message() -> None:
    """Shows a help message with usage instructions"""
    print(
        Fore.CYAN
        + "Converts PyTorch model files (.pt, .pth) to the safer .safetensors format.\n"
    )
    print(Fore.CYAN + Style.BRIGHT + "Usage:")
    print(
        Fore.CYAN
        + "python "
        + Style.BRIGHT
        + "safetensors_converter.py <input file/folder>"
        + Style.NORMAL
        + " [output folder] [--verbose]\n"
    )
    print(Fore.CYAN + Style.BRIGHT + "Arguments:")
    print(
        Fore.CYAN
        + Style.BRIGHT
        + "* input file/folder:"
        + Style.NORMAL
        + "\n\tRequired.\n\tEither a single file or a folder containing .pt or .pth files to convert."
    )
    print(
        Fore.CYAN
        + Style.BRIGHT
        + "* output folder:"
        + Style.NORMAL
        + "\n\tOptional.\n\tThe folder to save the converted files to.\n\tDefault: Subfolder 'converted_safetensors', created inside the input folder/file's folder."
    )
    print(
        Fore.CYAN
        + Style.BRIGHT
        + "* --verbose:"
        + Style.NORMAL
        + "\n\tOptional.\n\tWill print extra details for each model file during the process.\n\tDefault: Off - Will only print basic info during the process.\n"
    )
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + "\n* Important note !\n"
        + Style.NORMAL
        + "* Some models do not work when converted to .safetensors format.\n"
        + Style.BRIGHT
        + "* Test the produced .safetensors files before deleting the original files!\n"
        + Style.NORMAL
        + "* The script won't delete or otherwise modify the original files.\n"
    )
    return None


def check_safetensors_version() -> Union[bool, None]:
    """Checks if the installed safetensors version is at least 0.4.1 (required for handling files larger than 4 GB) and asks user what to do"""
    safetensors_version = importlib.metadata.version("safetensors")

    if version.parse(safetensors_version) < version.parse("0.4.1"):
        print(
            Fore.RED
            + Style.BRIGHT
            + f"\n* Warning *\nYour current safetensors library version ({safetensors_version}) can only handle models smaller than 4 GB.\n"
        )
        print(
            Fore.RED
            + "It will still work correctly, but models larger than 4 GB will be skipped.\nYou can proceed if this limitation doesn't affect you, otherwise exit and update the safetensors library.\n"
        )
        user_input = input(
            Fore.RED
            + f"Continue with current version {safetensors_version} and skip models larger than 4 GB? y/[n] :: "
        )
        if user_input.lower().strip() != "y":
            print(
                Fore.YELLOW
                + "\n* Update to the latest safetensors version, or at least install version 0.4.1 or higher\n* (e.g. with 'pip install \"safetensors>=0.4.1\"' or other appropriate way for your system).\n\n"
                + Fore.RED
                + "Exiting ...\n"
            )
            return None
        else:
            print(
                Fore.YELLOW
                + f"\n* Continuing with current safetensors version {safetensors_version} :: Skipping models larger than 4 GB.\n"
            )
            return False
    else:
        print(
            Fore.GREEN
            + f"* The installed safetensors library version ({safetensors_version}) supports files > 4 GB :: No model memory size limitations.\n"
        )
        return True


def get_model_mem_size(state_dict: dict, less_than_gb: int = 0) -> Union[int, bool]:
    """Calculates the memory size of a state dict and optionally checks if it's smaller than the given limit"""
    # Checking file size isn't accurate really, we need the actual memory size
    total_size = sum(
        t.numel() * t.element_size()
        for t in state_dict.values()
        if isinstance(t, torch.Tensor)
    )
    if less_than_gb == 0:
        return total_size
    return total_size < less_than_gb * (1024**3)


def print_results(
    summarized: Dict[str, int],
) -> None:
    """Prints the results of the conversion process"""
    # Print the summary
    print(Fore.CYAN + Style.BRIGHT + "\n- Results: -" + "-" * 48)
    for key, value in summarized.items():
        if value > 0:
            message_color = Fore.RED if "Failed" in key else Fore.GREEN
            print(message_color + f"* {value} {key}")
    print(Fore.CYAN + Style.NORMAL + "-" * 60 + "\n")


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> OrderedDict[str, Any]:
    """Self-explanatory, flattens nested, multi-level dicts to single-level ones"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, OrderedDict)):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)


def convert_to_float32(
    tensor: Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[str, Any]]
) -> Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[str, Any]]:
    """Converts all tensor types to float32"""
    if isinstance(tensor, torch.Tensor):
        return tensor.float().contiguous()
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(convert_to_float32(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: convert_to_float32(v) for k, v in tensor.items()}
    return tensor


def get_state_dict(
    checkpoint: Union[torch.nn.Module, Dict[str, Any]]
) -> Dict[str, Any]:
    """Gets a checkpoint's state dict"""
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint)
    raise ValueError("Unsupported checkpoint format")


def convert_to_safetensors(
    input_file: str, output_file: str, ok_sft_version: bool
) -> str:
    """Conversion function, receives a model file, converts it to .safetensors and saves it"""
    # Load the file and prepare the state dict
    try:
        state_dict = get_state_dict(
            torch.load(input_file, map_location="cpu", weights_only=True)
        )
    except Exception as e:
        # Is it because of weights_only=True ?
        if "Weights only load failed" in str(e):
            initial_e = str(e)
            # Ask user if it's ok to try again with weights_only=False
            print(Fore.RED + Style.BRIGHT + f"  Failed\n{e}")
            print(
                Fore.YELLOW
                + Style.BRIGHT
                + "\n* Info - please read:\n* The model failed to load with the default 'torch.load' option 'weights_only=True'\n* You can retry loading the model but this time with 'weights_only=False'."
            )
            print(
                Fore.YELLOW
                + "* This has the risk of allowing execution of any potentially malicious code inside the model file!\n* Before retrying make sure the model file comes from a relatively trustworthy source."
            )
            print(
                Fore.YELLOW
                + "* Usually, popular sites like HuggingFace, CivitAI, OpenModelDB and GitHub scan their files and are trustworthy.\n* The responsibility is still yours though, so if you rather not take the risk then please don't!\n"
            )
            user_input = input(
                Fore.YELLOW
                + "Accept the risk and retry with 'weights_only=False'? y/[n] :: "
            )
            if user_input.lower().strip() != "y":
                return f"2_**_Could not load the input file with weights_only=True"
            try:
                state_dict = get_state_dict(
                    torch.load(input_file, map_location="cpu", weights_only=False)
                )
            except Exception as e:
                return f"2_**_Could not load the input file and retrieve its state dict despite trying with both options of weights_only:\nweights_only=True:\n{initial_e}\nweights_only=False:\n{e}"
        else:
            return (
                f"2_**_Could not load the input file and retrieve its state dict:\n{e}"
            )
    processed_state_dict = convert_to_float32(flatten_dict(state_dict))

    if not ok_sft_version:
        # Check if the model memory size is smaller than 4 GB
        if not get_model_mem_size()(processed_state_dict, less_than_gb=4):
            return "3_**_The model's memory size is larger than 4 GB, and the current safetensors version cannot handle such models"

    # Try saving
    try:
        save_file(processed_state_dict, output_file)
        # It just worked I guess
        return "0_**_Conversion successful"
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
                return "1_**_Conversion successful after non-contiguous tensors were found and made contiguous"
            except Exception as e:
                return f"4_**_Non-contiguous tensors were found and made contiguous, but the model conversion still failed:\n{e}"
        elif "invalid load key" in str(e):
            # Invalid/corrupted file
            return f"5_**_Invalid load key found while trying to convert the model (usually because of either invalid format or corrupted file):\n{e}"
        else:
            # Return the original exception since it's not one of the known ones
            return f"6_**_Conversion failed for other reason:\n{e}"


def main_processor(
    input_folder: str,
    input_file: str | None,
    output_folder: str,
    ok_sft_version: bool,
    verbose: bool,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, Union[int, str]]]]:
    """Main processing function, handles the input(s) and calls the conversion function as necessary"""
    # Get the list of files to process
    files_to_process = []
    if input_file is not None:
        files_to_process.append(input_file)
    else:
        files_to_process = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
            and f.endswith((".pt", ".pth"))
        ]

    print(
        Fore.CYAN
        + (
            f"\n* Found {len(files_to_process)} .pth and .pt model files to convert:"
            if verbose
            else f"\n* Processing {len(files_to_process)} models:"
        )
    )
    # Result structures
    detailed_results = {}
    result_codes = [
        "model files were processed in total",
        "models were converted successfully",
        "models were converted successfully after making all their tensors contiguous",
        "models failed while trying to load with torch.load and retrieve their state dict",
        "models were skipped because of safetensors' 4 GB memory size limitation",
        "models failed to convert despite making all found non-contiguous tensors contiguous",
        "models failed to convert because they had invalid format or corrupted data",
        "models failed because of other reasons not covered above",
    ]

    summarized_results = {key: 0 for key in result_codes}
    summarized_results["model files were processed in total"] = len(files_to_process)

    # Iterate over the file list
    for i, file in enumerate(files_to_process):
        print(
            Fore.CYAN
            + f"\n- {str(i+1).zfill(3)} / {str(len(files_to_process)).zfill(3)} -",
            end="",
        )
        if verbose:
            print(Fore.CYAN + f"\n[{os.path.basename(file)}] :::", end="")
        output_file = os.path.join(
            output_folder,
            os.path.basename(file)
            .replace(".pth", ".safetensors")
            .replace(".pt", ".safetensors"),
        )
        file_result = convert_to_safetensors(file, output_file, ok_sft_version).split(
            "_**_"
        )
        detailed_results[file] = {
            "result_code": int(file_result[0]),
            "result_message": file_result[1],
        }
        if detailed_results[file]["result_code"] <= 1:
            detailed_results[file]["output_file"] = output_file
            result_display_color = Fore.GREEN
            result_short_message = "OK"
        else:
            detailed_results[file]["output_file"] = ""
            result_display_color = Fore.RED
            result_short_message = "Failed"
        if verbose:
            print(
                result_display_color
                + Style.BRIGHT
                + f"  {result_short_message}\n"
                + Style.NORMAL
                + f"{detailed_results[file]['result_message']}"
            )
        else:
            print(result_display_color + Style.BRIGHT + f"  {result_short_message}")
        summarized_results[result_codes[detailed_results[file]["result_code"] + 1]] += 1
    return summarized_results, detailed_results


if __name__ == "__main__":
    init(autoreset=True)
    print(
        Fore.CYAN
        + "\n "
        + "-" * 39
        + "\n--| "
        + Style.BRIGHT
        + "SafeTensors Converter Script - v2"
        + Style.NORMAL
        + " |--\n"
        + " "
        + "-" * 39
        + "\n\n"
    )

    # Show help message and exit if no args or if --help/-h is passed
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("--help", "-h")):
        show_help_message()
        sys.exit(0)

    # Parse the command line arguments
    parsed_args = parse_args(sys.argv)
    if parsed_args is None:
        sys.exit(1)
    else:
        input_folder, input_file, output_path, verbose_mode = parsed_args

    # Try creating the output folder if it doesn't exist
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(
            Fore.RED
            + Style.BRIGHT
            + "\n\nError! "
            + Style.NORMAL
            + f"Trying to create the output folder ({output_path}) failed:\n{e}\nResolve or specify other output folder. Exiting ...\n"
        )
        sys.exit(1)

    # Check safetensors version
    if verbose_mode:
        print(Fore.CYAN + "* Checking the installed safetensors library version ...")

    ok_sft_version = check_safetensors_version()
    if ok_sft_version is None:
        sys.exit(1)

    # Process and get results
    if verbose_mode:
        print(
            Fore.CYAN
            + "* Starting the conversion process with the following parameters:"
        )
        if input_file:
            print(Fore.CYAN + f" + Run mode:\tFile\n" + f" + Input:\t{input_file}")
        else:
            print(Fore.CYAN + f" + Run mode:\tFolder\n" + f" + Input:\t{input_folder}")
        print(Fore.CYAN + f" + Output:\t{output_path}")
        print(Fore.CYAN + f" + Size limit:\t{not ok_sft_version}")
        print(Fore.CYAN + f" + Verbose:\t{verbose_mode}")

    summarized_results, detailed_results = main_processor(
        input_folder, input_file, output_path, ok_sft_version, verbose_mode
    )

    print(Fore.GREEN + Style.BRIGHT + "\n=| Conversion Completed |=\n")

    # Print results and save them as json to the output folder
    print_results(summarized_results)
    json_filename = os.path.join(
        output_path, f"_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    with open(json_filename, "w") as json_file:
        json.dump(detailed_results, json_file, indent=2, ensure_ascii=False)

    # Print footer and exit
    print(Fore.CYAN + Style.BRIGHT + "\n- Notice: -" + "-" * 49)
    print(
        Fore.CYAN
        + f"The converted safetensor files can be found in:\n"
        + Style.BRIGHT
        + f"{output_path}"
    )
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + "Remember to test them "
        + Fore.CYAN
        + Style.NORMAL
        + "before deleting their original files!\n"
    )
    print(
        Fore.CYAN
        + f"A JSON file with the individual results for each model file\nwas also saved to the output folder above.\n"
        + "-" * 60
        + "\n"
    )

    sys.exit(0)
