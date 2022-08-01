# Data-Inference
## Install the environment (Python 3.9.0)

First you will create a virtual environment with the following commands:

*Linux/MacOS*

```console
python3 -m venv <DIR>
```

*Windows*

```console
py -m venv <virtual_environment_directory>
```

Then, activate your virtual environment:

*Linux/Macos*

```console
source <DIR>/bin/activate
```

*Windows*

```console
<virtual_environment_directory>\Scripts\activate
```

Finally, install the required libraries:

*Linux/Macos*

```console
python3 -m pip install -r requirements.txt
```

*Windows*

```console
py -m pip install -r requirements.txt
```

## Use our tools

This is the structure of our project:

```
.
├── Inference/
│   ├── KnowledgeDistillation/
│   │   ├── CCKD (Correlation Congruence for Knowledge Distillation)
│   │   ├── CKD (Classic Knowledge Distillation)
│   │   └── SPKD (Similarity-Preserving Knowledge Distillation)
│   └── Quantization
└── Data
```

In each sub-part of the project, you will find a README.md which will help you to use our tools thanks to documentation and examples.