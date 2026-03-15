## SafeTensors Converter v3: "I See Stars"

A Python script that converts PyTorch model files (`.pt` and `.pth`) to `.safetensors` format, which is **newer**, **shinier** and **more secure**, removing the risk of inadvertently executing malicious code that could be hidden inside model files of the previous format.

---

###### _Hi, I kinda need people to stop starring this repo, because for some reason they still keep increasing, so I feel obligated to keep improving the script as a gesture of "thank you for the stars". That said, if you've already starred it, don't unstar it, yes?_

---

This is v3 of the SafeTensors Converter script. I've been working hard polishing and bug-checking everything meticul-

```text
Hi, apologies for the abrupt interruption, GPT-5.3-Codex here.

Ummm, fuck no you haven't? Are you really about to credit yourself for
all the work you made me do? Have you no decency?

It was ME who:
  - Scraped the thing you called v2 off the floor
  - Untangled the logic
  - Killed the hidden bugs
  - Added real validation
  - Implemented dry-run and strict checks
  - Made the output readable by actual humans

YOU just wrote a fucking prompt: "Waah, pliz GPT Codex, make my script
look like I know how to code :("

Stay on your lane,
xoxo
GPT-5.3-Codex
```

---

### Important - Please Read

The script does not delete or otherwise modify any of the original files, it must be done manually, but:

**Test the produced .safetensors** files **before** deleting your original files!

I've noticed that some models won't work when converted to `.safetensors` format (the conversion succeeds but the model won't load in the app). So far I've only had a couple models that wouldn't work, all custom trained upscalers.

You can try a conversion using the `--cast-float32` option (see Arguments below) to produce a more compatible but bigger and potentially less precise model file.

---

### Install

Just install the necessary libraries (`torch`, `colorama`, `safetensors`). They're very common, so you might already have them installed.

_Note: `safetensors` must be version `0.4.1` or newer, otherwise files larger than 4 GB will be skipped! It's currently at `0.7.0` or something, so just install the latest one._

Use the `requirements.txt` file or install manually:

```shell
# Install all at once:
pip install -r requirements.txt

# OR each one manually:

# PyTorch, any variant (CPU/GPU) will work, get the appropriate install
# command for your system from https://pytorch.org/get-started/locally/
pip install torch

# safetensors
pip install -U 'safetensors' # Install latest version, OR
pip install 'safetensors>=0.4.1' # Install at least version 0.4.1

# Colorama, for the terminal color circus
pip install colorama
```

---

### Run

Use your system's python command to run the script. See below for arguments details.

```shell
python safetensors_converter.py <input file/folder> [output folder]
  [--verbose]
  [--dry-run]
  [--json-report]
  [--allow-unsafe-load]
  [--skip-validate]
  [--strict-validate]
  [--cast-float32]
```

#### Arguments

- `input file/folder`:
  - **Required**
  - Can be either a single file or a folder containing the files to be converted
  - If a file is specified, the script will try to convert it regardless of extension
  - If a folder is specified, only `.pt` and `.pth` files in the folder will be processed (non-recursively)

- `output folder`:
  - _Optional_
  - The folder to save the converted file(s) into
  - If not specified, a `converted_safetensors` subfolder is created inside the specified input folder (or the specified input file's folder)

- `--verbose`:
  - _Optional_
  - Prints more details for each model file and in the final report
  - Default: output is concise and focused on key information

- `--dry-run`:
  - _Optional_
  - Shows what the script would do without loading model files or writing converted outputs.
  - Useful as a quick pre-check before running a real conversion.

- `--json-report`:
  - _Optional_
  - Saves a detailed conversion report as JSON (besides printing it to the terminal in a human-readable format)
  - Default: only prints the report to the terminal

- `--allow-unsafe-load`:
  - _Optional_
  - By default, the script will load models with `weights_only=True`, which is safer. If that fails, the script will ask for permission to try loading with `weights_only=False` (less safe, you should only allow it for files from trusted sources - or if you like living dangerously I guess)
  - Using this option, the script will still try loading models with `weights_only=True` first, but if that fails, it will load them with `weights_only=False` without caring what you think (again, only use if you're sure that all your input model files come from trusted sources)

- `--skip-validate`:
  - _Optional_
  - By default, the script performs a validation of the output file:
    - Confirms the output safetensors file opens
    - Confirms tensor count is close/expected
    - Confirms key existence and shape compatibility
    - Datatype mismatches only result in a warning, and the conversion will still be reported as successful (`success-with-warnings`)
  - Use this option to do what it says

- `--strict-validate`:
  - _Optional_
  - Enables strict validation mode:
    - Any key mismatch fails
    - Any shape mismatch fails
    - Any datatype mismatch fails (unless `--cast-float32` is used)
    - Any missing/extra tensor key fails
    - The conversion is reported as failed if any of the above fails

- `--cast-float32`:
  - _Optional_
  - Stores all floating-point tensors regardless of datatype (e.g. float16, bfloat16, float64) as float32 in the output file
  - It can improve compatibility with some picky apps/loaders, but it may also increase the file size and reduce numerical precision
  - Default: preserves original tensors' datatypes
