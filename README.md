### safetensors converter script

*(i mean it's probably the 100000th one in github, but whatever)*

A lil' script that took much more time than what I'd like to admit, and that converts pickled Python files to safetensors format. It accepts a folder as an argument and converts every pth file in it (or pt, or other user-specified extension). It doesn't delete the original files, and it puts the safetensors files in a new subfolder it creates.


###### Usage:

```shell
python safetensors_converter.py <input-folder> [file-types-to-convert (default: pth)]
```

`input-folder` : the folder containing the files to convert (required)

`file-types-to-convert` : extension of the files to convert (optional, `pth` is default)

Requires [safetensors](https://pypi.org/project/safetensors/ "safetensors package page") and [torch](https://pytorch.org/get-started/locally/ "pytorch download page") Python packages installed.


###### Important notice:

So far the script has been working fine and creating functioning safetensor files, but I've encountered some pth files that wouldn't work when converted to safetensors format. It's mostly been custom made upscale models.

The script doesn't delete any files, so before deleting your original pth file make sure the safetensors version works.


###### ToDo:

* Not much development potential here tbh :D Maybe I'll create a Gradio interface when I'm in the mood for it?
