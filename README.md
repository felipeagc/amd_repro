## Build instructions

```
cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug .
cmake --build build
./build/repro # run it
```

## Results from Linux

This is using Linux kernel 5.9 and the RADV vulkan driver.

Sometimes I get a completely frozen desktop, sometimes I get `VK_ERROR_DEVICE_LOST` (error code -4).

Here's an instance of `VK_ERROR_DEVICE_LOST` (there is no desktop freeze):

```
Using physical device: AMD RADV NAVI14 (ACO)
amdgpu: The CS has been rejected, see dmesg for more information (-2).
../rendergraph.c:5608 vulkan error: -4
```

dmesg says:

```
[  487.910885] [drm:amdgpu_cs_ioctl [amdgpu]] *ERROR* Failed to initialize parser -2!
```
