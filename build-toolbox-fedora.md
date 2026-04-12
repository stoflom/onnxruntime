# Build Instructions

To build ONNX Runtime with AMD ROCm 7.2 on Fedora 43 using AMD-supplied RPMs. See [AMD ROCm Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

## Create Environment

Create a toolbox for Fedora and enter it.

```bash
toolbox --help
```

## Update System

```bash
sudo dnf -y update
sudo dnf group install "Development Tools"
sudo dnf install cmake patch gmock-devel
sudo dnf install python3 python3-setuptools python3-wheel
sudo dnf install python3-devel python3-httpx python3-pyyaml \
  python3-typing-extensions
```

## Install ROCm Repositories

Install ROCm repositories as per AMD instructions for RHEL 10.1:

```bash
sudo dnf install https://repo.radeon.com/amdgpu-install/7.2.1/rhel/10/amdgpu-install-7.2.1.70201-1.el10.noarch.rpm
sudo dnf clean all
```

## Configure Repositories

```bash
# Delete amdgpu.repo (cannot load drivers in toolbox)
sudo rm /etc/yum.repos.d/amdgpu.repo
```

**Edit /etc/yum.repos.d/rocm.repo to remove amdgraphics (ditto)**

## Install ROCm

```bash
sudo dnf clean all
sudo dnf install rocm
```

## Give User Rights to GPU Hardware

```bash
# Add the current user to the render and video groups
sudo usermod -a -G render,video $USER
```

## Build ONNX Runtime

Run the build script with specific flags:

```bash
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --use_migraphx --build_wheel
```

## Install Binaries under /usr

```bash
cd build/Linux/Release && sudo make install
```

## Install Python Wheel

Create a virtual python environment  (or your on-host python libs will be clobbered)

```bash
#The following will create the .venv under build/Linux/Release
python3 -m venv .venv
source .venv/bin/activate
pip install build/Linux/Release/dist/onnxruntime*.whl --force-reinstall
```
Run the python test script below inside the virtual environment.

## Test Installation

```bash
python test-gpu.py
```

**(Note: Edit the file first to use your available ONNX file)**

### Expected Output

Both the MIGraphX- and the CPU execution providers should be listed.

```text
Active provider: ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

To deactivate the python virtual environment do:

```bash
deactivate
```
