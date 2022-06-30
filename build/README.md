
# Build

```bash
cmake -DCMAKE_PREFIX_PATH=/path/path/libtorch-sample/libtorch/ ..
make -j4
```

# Run

```bash
./example-app ../traced_resnet_model.pt ../cat.png
```
