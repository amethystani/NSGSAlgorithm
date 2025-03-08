// This file is not meant to be compiled, it's just instructions on how to fix the include path issue

/*
To fix the include path issue, you need to copy the required header files from the ONNX Runtime include directory
to your project's include directory.

Run the following commands from the Backend directory:

cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_cxx_api.h include/
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_c_api.h include/
cp -f onnxruntime-osx-arm64-1.15.0/include/onnxruntime_cxx_inline.h include/

These commands will copy the necessary header files to your project's include directory,
allowing the compiler to find them when using #include "onnxruntime_cxx_api.h".
*/ 