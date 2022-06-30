
# Install libtorch

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
# Create traced model

```bash
python main.py
```
# Build C++ code & Run

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/.../pytorch2cpp/libtorch ..
make -j4
```
Ref : [Link](./build/README.md)

