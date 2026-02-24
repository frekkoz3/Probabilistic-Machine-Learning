# Probabilistic Machine Learning

This repo contains the content of the "Probabilistic Machine Learning" academic course, which is part of the wider course "Deep and Probabilistic Machine Learning".

Lecturer: Prof. Luca Bortolussi Tutors: Sara Candussio, Alessandro Della Siega

The material comes from [this repository](https://github.com/gaoithee/PML26) and the official Team group of the course.

---

## Structure

```bash
├── lectures    # folder containing lectures slides
├── notes       # folder containing hand notes for the lectures
└── exercises   # folder containing python and notebook for the exercises
```

---

## Setup

The quickest and simplest way to set up everything is through the **Jupyter VS Code "Create Environment"** option.

Follow the steps below carefully.

---

## Step-by-Step Procedure (VS Code + Jupyter)

### Open the Project in VS Code

- Open **Visual Studio Code**
- Click on **File → Open Folder**
- Select the root folder of this project

Make sure the folder contains the `requirements.txt` file.

---

### Open a Jupyter Notebook (or Create One)

- Open an existing `.ipynb` file  
  OR  
- Create a new one:
  - `Ctrl + Shift + P`
  - Search for: `Create: New Jupyter Notebook`

---

### Select / Create a Python Environment

At the top-right of the notebook, click on the **Kernel Selector**.

Then:

- Click **"Select Kernel"**
- Choose **"Python Environments..."**
- Click **"Create Python Environment"**

---

### Choose Environment Type

When prompted:

- Select **Virtual Environment (venv)**

---

### Select Python Interpreter

Choose a Python version (recommended: **Python 3.9+**).

If you don’t see one:

- Install Python from https://www.python.org/downloads/

---

### Install Dependencies Automatically

When VS Code detects the `requirements.txt` file, it will ask:

> "Install packages from requirements.txt?"

Click **Yes**.

VS Code will automatically:

- Create the virtual environment
- Install all dependencies listed in `requirements.txt`
- Configure the kernel for Jupyter

---

### Verify Installation

Run the following in a notebook cell:

```python
import sys
print(sys.executable)
```

---

## ⚠️ Troubleshooting

- **Broken System Python:** If you work with `venv`s and see `ModuleNotFoundError: No module named 'encodings'`, your system's Python 3.12 installation is incomplete. You can switch to `conda` installation to avoid this issue.
- **PyTorch & GPU:** If you have an NVIDIA GPU, the standard `pip install` might only install the CPU version. Visit [pytorch.org](https://pytorch.org/) for the specific CUDA-enabled command.
- **Windows C++ Tools:** If `pip install` fails on packages like `scipy`, you may need the [Microsoft Visual C++ Build Tools](https://www.google.com/search?q=https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## ⚠️ Note

This serve merely as wrapper of all the works produced by the professors and the tutor in order to help the ones having problems accessing the materials on other platforms.
