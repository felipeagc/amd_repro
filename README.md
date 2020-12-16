## Description

I'm getting some crashes when using two renderpasses on an AMD GPU:
I draw some content to an image in renderpass 1 and sample it in renderpass 2,
rendering it to a swapchain image, all within one command buffer submission.

The main code is in `main.c`, with my vulkan abstraction layer in `rendergraph.c`.

## Build instructions

```
cmake -Bbuild .
cmake --build build
./build/repro # run it (linux)
.\build\Debug\repro.exe # run it (windows)
```

## Testing specs

I'm getting these crashes on an RX 5500 XT 8GB. 

With validation layers enabled, there are no warnings or error messages, and, curiously, the program does not crash (on both Windows and Linux).
The crashes only happen with validation layers disabled.

I've also tested on my Intel laptop and it works fine there.
It also works fine with my other Nvidia GPU.

## Results from Linux

This is using Linux kernel 5.9 and the RADV vulkan driver, but it also happens with AMDVLK.

Most times I get a completely frozen desktop, sometimes I get `VK_ERROR_DEVICE_LOST` (error code -4).

Here's an instance of `VK_ERROR_DEVICE_LOST` (there is no desktop lockup):

```
Using physical device: AMD RADV NAVI14 (ACO)
amdgpu: The CS has been rejected, see dmesg for more information (-2).
../rendergraph.c:5608 vulkan error: -4
```

dmesg says:

```
[  487.910885] [drm:amdgpu_cs_ioctl [amdgpu]] *ERROR* Failed to initialize parser -2!
```

## Results from Windows

This is on Windows 10, using Radeon Software version 20.11.3. Radeon Software also gives me these versions:

```
Software Version 2020.1125.1407.25415
Driver Version   20.45.01.24-201125a-361677E-RadeonSoftwareAdrenalin2020
Vulkan™ Driver   Version 2.0.168
```

On Windows, the crash happens only when resizing the window, sometimes taking 2 resizes for it to crash.

Most times I get `VK_ERROR_DEVICE_LOST`:

```
Using physical device: Radeon RX 5500 XT
F:\code\c\amd_repro\rendergraph.c:5612 vulkan error: -4
```

But other times I'll get a driver timeout or a BSOD.
