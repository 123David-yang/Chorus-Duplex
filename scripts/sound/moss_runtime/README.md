# MOSS ONNX runtime

This directory contains the lightweight ONNX runtime files from
[OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano), revision
`cc7bdf19c7639c0870dab22045a33b442760f6be`, distributed under the accompanying
Apache-2.0 license.

`onnx_tts_runtime.py` is adapted to use SoundFile instead of importing
PyTorch/torchaudio and to avoid requiring the upstream Python package for its
default output directory. Its reference-audio loader handles NumPy arrays and
resampling without PyTorch. Chorus-Duplex also provides a reference-audio
loading adapter in `scripts/sound/synthesis/backends.py`. Model weights are
not included; obtain them from the upstream model repositories.
