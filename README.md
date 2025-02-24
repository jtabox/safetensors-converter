### SafeTensors converter script v2 - \o/

*(yes, the 10000th one on github but whatever)*

A Python script that converts PyTorch model files and checkpoints (.pt and .pth) to the .safetensors format which is newer, shinier and is a more secure way to store and load model data, without the risk of potentially executing malicious code hidden inside a model file of the previous format.

### "Did it really need a v2 lol"

Alright. So, for some incomprehensible to me reason, this repo has received 10 stars, which is about 10 more than what I ever expected it (or me in general) to receive. So I kinda felt obligated to say "thanks for the stars", and I decided to do it by writing a (hopefully) better version of the script.
~~Of course there's also the very practical matter of the previous script not supporting files larger than 4 GB because it was using an older version of the safetensors library, which was a major motive, ngl.~~
Changes from v1:

* The biggest one,this version **supports converting model files larger than 4 GB**
* The script can optionally target specific files now, not just all files in a folder
* It no longer accepts a file extension as an argument. If given a folder it will convert all .pt and .pth files in it (felt kinda weird to have that option to begin with tbh)
* Will now try by default to load a model file using `weights_only=True`, which is safer than the previous iteration. If that fails though, it'll ask the user if they want to try loading the file with `weights_only=False`
* The information displayed throughout the process and after it is a bit more thought out than dumping logs on the screen
* It will automatically write a JSON file with more details about each file's conversion results at the end of the process
* **C O L O R**. In the terminal. Lots of colored text. It's like a circus now. I don't care.
* Other minor silly stuff that showed my lack of experience in Python and in coding in general. I've fixed those, but I'm sure I'll find double so many when I read through the script again in 6 months, but what can you do...

### Installation & Usage

* **Install**:
  No installation required for the script itself, but it uses 3 packages that might need to be installed if they're not already present.
  There's a requirements.txt file available to use, otherwise they can be installed manually:

```shell
# Install all at once
pip install -r requirements.txt

# OR each one manually
pip install 'torch' # No specific variant required (CPU/GPU)

# safetensors version 0.4.1 or newer is required, otherwise files larger than 4 GB will be skipped!
pip install -U 'safetensors' # Latest version is recommended, otherwise use: pip install 'safetensors==0.4.1'

pip install colorama # For the terminal circus
```

* **Run**:

```shell
python safetensors_converter.py <input_file/folder> [output_folder] [--verbose]
```

* **Arguments**:

```
- input_file/folder:
  Required.
  Can be either a single file or a folder containing .pt and .pth files to convert.
  If it's a file, it'll be converted regardless of its extension.

- output_folder:
  Optional.
  The folder to save the converted file(s) to.
  If not specified, a 'converted_safetensors' subfolder is created inside the input folder or the input file's folder.

- --verbose:
  Optional.
  Will print some extra details for each model file during the process.
  If not specified, the script will only print basic info during the process.
  In any case, a JSON file with more info will be saved in the output folder.
```

### Important Note - Please Read

Make sure you **test the produced .safetensors** files *before deleting your original files*!

* Some models won't work when converted to .safetensors format, depending on the model and the app using it.
* So far I've mostly encountered a couple models that wouldn't work, they were all custom trained upscalers. They were successfully converted, but the app using them would complain and not load them.
* The script won't delete or otherwise modify any of the original files anyway, so you can always go back to them if needed or manually delete them if your .safetensors versions work.
