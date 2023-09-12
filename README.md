## OSSL-pydev

Python package for processing [Open Soil Spectral Library (OSSL)](https://explorer.soilspectroscopy.org) and other spectral data using the xSpectre json command and file structure. The top folder of the repo contains 2 files and 2 folders. The 2 files are this README.md file and the LICENCE file.

## Licence

The licence is a BSD 3-Clause, and the text is replicated below:

```
BSD 3-Clause License

Copyright (c) 2023, Karttur

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## anaconda

The folder _anaconda_ contains two files:
- .condarc
- spectra-ossl_from-history_py38.yml

These files can be used for setting up a virtual python environment using the python package manager [Anaconda](https://anaconda.org). How to install and setup Anaconda and then create the virtual python environment for the python package in this repo (ossl-xspectre) is covered in the blog [xSpectre soil spectral libraries](https://karttur.github.io/soil-spectro/), specifically in the post [Anaconda](http://localhost:4000/libspectrosupport/spectrosupport-OSSL-anaconda/).

## ossl-xspectre

The python package in this repo is called _ossl-xspectre_ and contains 4 python (.py) modules:
- OSSL_import.py
- OSSL-plot.py
- OSSL-mlmodel.py
- xSpectre_Import.py

The modules are setup for a virtual python environment with the first line of each module pointing towards this environment:

```
#!/Applications/anaconda3/envs/spectraimagine_py38/bin/python3.8
```
This need to be edited to reflect your python setup.

Additionally the folder with the package contains two extra files:
- .project
- .pydevproject

These two files belong to the IDE [Eclipse](https://www.eclipse.org) and the package _ossl-xspectre_ can be directly imported to Eclipse.

How to install and setup Eclipse and then import and run ossl-xspectre is covered in the blog [xSpectre soil spectral libraries](https://karttur.github.io/soil-spectro/), specifically in the post [Eclipse for PyDev](http://localhost:4000/libspectrosupport/spectrosupport-OSSL-eclipse/).

## Running the ossl-xpsectre modules

The modules of ossl-xspectre all require a specification of the local paths and names of 1) the OSSL data and 2) the command files that define how to do the processing:

- **rootpath**: full path to folder with a downloaded OSSL zip file; parent folder to  "sourcedatafolder", "arrangeddatafolder", and "jsonfolder"
- **sourcedatafolder**: subfolder under "rootpath" with the exploded content of the OSSL zip file (default = "data")
- **arrangeddatafolder**: subfolder under "rootpath" where the processed OSSL data will be stored
- **jsonfolder**: subfolder under "rootpath" where the json parameter file ("projFN") is located
- **projFN**: the name of an existing txt file that sequentially lists json parameter files to run, must be in the subfolder "jsonfolder"
- **createjsonparams**: if set to true the script will create a template json file and exit

```
{
  "rootpath": "/local/path/to/folder/with/OSSL/data",
  "sourcedatafolder": "data",
  "arrangeddatafolder": "arranged-data",
  "jsonfolder": "json-xyx",
  "projFN": "project-xyz.txt"
  "createjsonparams": false
}
```

The details for how to download, import, plot and model spectral data, with focus on OSSL, using the ossl-xspectre package is covered in the blog [xSpectre soil spectral libraries](https://karttur.github.io/soil-spectro/).
