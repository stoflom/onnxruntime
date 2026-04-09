# Build Instructions

To build onnxruntime with AMD ROCm 7.2 on Fedora 43 using AMD supplied rpms. See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html

## Create Environment

   Create a toolbox for fedora and enter it.

   see

   ```bash
   toolbox --help
   ```

## Update System**

 ```bash
   sudo dnf -y update
   sudo dnf group install c-development
   sudo dnf install cmake patch
   sudo dnf install python python3-setuptools python3-wheel
   sudo dnf install python-devel python3-httpx python3-pyyaml \
    python3-typing-extensions

 ```

## Install ROCm Repositories**

   Install ROCm repos as per AMD instructions for RedHat 10.1

```bash
   sudo dnf install https://repo.radeon.com/amdgpu-install/7.2.1/rhel/10/amdgpu-install-7.2.1.70201-1.el10.noarch.rpm
   sudo dnf clean all
```

## Configure Repositories**
  
   ```bash
   # Delete amdgpu.repo (cannot load drivers in toolbox)
   rm /etc/yum.repos.d/amdgpu.repo

   # Edit /etc/yum.rc.d/rocm.repo to delete amdgraphics (ditto)
   ```

## Install ROCm**

   ```bash
   sudo dnf clean all
   sudo dnf install rocm
   ```

## Give user rights to the GPU hardware

   ```bash
   sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
   ```

## Build onnxrunner**

   Run the build script with specific flags:

   ```bash
   ./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --use_migraphx --build_wheel
   ```

## Install binaries under /usr

   ```bash
      cd build/Linux/Release && sudo make install
   ```

## Install Python Wheel

   ```bash
   pip install build/Linux/Release/dist/onnxruntime*.whl --force-reinstall
   ```

## Test Installation

   ```bash
   python test-gpu.py
   ```

(Note: Edit the file first to use your available onnx file)

### Expected Output:

   ```text
   Active provider: ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
   ```
