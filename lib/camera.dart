import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'models.dart';

typedef void Callback(List<dynamic> list, int h, int w);

class Camera extends StatefulWidget {
  final List<CameraDescription> cameras;
  final Callback setRecognitions;
  final String model;

  Camera(this.cameras, this.model, this.setRecognitions);

  @override
  _CameraState createState() => new _CameraState();
}

class _CameraState extends State<Camera> {
  CameraController? controller;
  bool isDetecting = false;
  late Interpreter interpreter;

  @override
  void initState() {
    super.initState();
    initializeCamera();
  }

// In initializeCamera method
  Future<void> initializeCamera() async {
    if (widget.cameras.isEmpty) {
      print('No camera is found');
      return;
    }

    try {
      print('Loading model: ${widget.model}');

      final options = InterpreterOptions()
        ..threads = 4; // Update the number of threads

      if (widget.model == mobilenet) {
        interpreter = await Interpreter.fromAsset(
          'assets/mobilenet_v1_1.0_224.tflite',
          options: options,
        );
      } else if (widget.model == posenet) {
        interpreter = await Interpreter.fromAsset(
          'assets/posenet_mv1_075_float_from_checkpoints.tflite',
          options: options,
        );
      } else {
        interpreter = await Interpreter.fromAsset(
          widget.model == ssd
              ? 'assets/ssd_mobilenet.tflite'
              : 'assets/yolov2_tiny.tflite',
          options: options,
        );
      }

      // Print input and output shapes for debugging
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);

      print('Input shape: ${inputTensor.shape}');
      print('Input type: ${inputTensor.type}');
      print('Output shape: ${outputTensor.shape}');
      print('Output type: ${outputTensor.type}');

      // Validate input shape matches our preprocessing
      final inputSize = computeTensorSize(inputTensor.shape);
      if (inputSize != 1 * 224 * 224 * 3) {
        print(
            'Warning: Input tensor size ($inputSize) does not match expected size (${1 * 224 * 224 * 3})');
      }

      print('Model loaded successfully: ${widget.model}');

      // Initialize camera controller
      controller = CameraController(
        widget.cameras[0],
        ResolutionPreset.high,
        enableAudio: false,
      );

      await controller!.initialize();

      if (!mounted) return;

      setState(() {});

      controller!.startImageStream((CameraImage img) {
        if (!isDetecting) {
          isDetecting = true;
          processImage(img).then((_) {
            isDetecting = false;
          }).catchError((e) {
            print('Error in image stream: $e');
            isDetecting = false;
          });
        }
      });
    } catch (e) {
      print('Error in initializeCamera: $e');
      return;
    }
  }

  // Update the processImage and preprocessImage methods:

  Future<void> processImage(CameraImage image) async {
    try {
      // Get the preprocessed input data
      var input = preprocessImage(image);

      // Get input and output tensors
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);

      // Print shapes for debugging
      print('Input tensor shape: ${inputTensor.shape}');
      print('Output tensor shape: ${outputTensor.shape}');

      // Calculate total input size
      final inputSize = inputTensor.shape.reduce((a, b) => a * b);

      // Create input buffer with proper shape
      var inputArray = Float32List(inputSize);
      var inputIndex = 0;

      // Fill the input array while preserving the 4D structure
      for (int b = 0; b < 1; b++) {
        for (int h = 0; h < 337; h++) {
          for (int w = 0; w < 337; w++) {
            for (int c = 0; c < 3; c++) {
              inputArray[inputIndex++] = input[b][h][w][c];
            }
          }
        }
      }

      // Create output buffer matching the expected output shape
      final outputSize =
          outputTensor.shape.reduce((a, b) => a * b); // 1 * 22 * 22 * 17
      final outputArray = Float32List(outputSize);

      // Allocate tensors
      interpreter.allocateTensors();

      // Run inference with reshaped tensors
      final inputs = [inputArray];
      final outputs = {0: outputArray};

      interpreter.runForMultipleInputs(inputs, outputs);

      // Process the output...
      final List<dynamic> results = processOutput(outputArray.toList());

      if (results.isNotEmpty) {
        widget.setRecognitions(results, image.height, image.width);
      }
    } catch (e, stackTrace) {
      print('Error during image processing: $e');
      print('Stack trace: $stackTrace');
    }
  }

  List<List<List<List<double>>>> preprocessImage(CameraImage image) {
    // Convert YUV420 to RGB
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    // Create buffer for RGB output
    final img.Image rgbImage = img.Image(width: width, height: height);

    // Convert YUV to RGB
    for (int y = 0; y < height; y++) {
      int pY = y * image.planes[0].bytesPerRow;
      int pUV = (y ~/ 2) * uvRowStride;

      for (int x = 0; x < width; x++) {
        final int uvOffset = pUV + (x ~/ 2) * uvPixelStride;

        // Y plane
        final int yValue = image.planes[0].bytes[pY + x] & 0xff;
        // U plane
        final int uValue = image.planes[1].bytes[uvOffset] & 0xff;
        // V plane
        final int vValue = image.planes[2].bytes[uvOffset] & 0xff;

        // YUV to RGB conversion
        int r = (yValue + 1.402 * (vValue - 128)).toInt().clamp(0, 255);
        int g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
            .toInt()
            .clamp(0, 255);
        int b = (yValue + 1.772 * (uValue - 128)).toInt().clamp(0, 255);

        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }

    // Resize to exactly 337x337 as required by the model
    final img.Image resizedImage = img.copyResize(rgbImage,
        width: 337, height: 337, interpolation: img.Interpolation.linear);

    // Create 4D input tensor [1][337][337][3]
    List<List<List<List<double>>>> input = List.generate(
      1, // batch size
      (i) => List.generate(
        337, // height
        (y) => List.generate(
          337, // width
          (x) {
            final pixel = resizedImage.getPixel(x, y);
            // Normalize pixel values to [0,1]
            return [
              pixel.r.toDouble() / 255.0,
              pixel.g.toDouble() / 255.0,
              pixel.b.toDouble() / 255.0
            ];
          },
        ),
      ),
    );

    return input;
  }

// Helper function to compute total size of tensor shape
  int computeTensorSize(List<int> shape) {
    return shape.reduce((a, b) => a * b);
  }

// Also update the processOutput method to handle the 22x22x17 output:

  List<dynamic> processOutput(List<double> output) {
    // Reshape the flat output array to [1][22][22][17]
    List<dynamic> results = [];

    try {
      // Process based on your model's specific output format
      // This is a placeholder - update according to your model's output structure
      int index = 0;
      for (int y = 0; y < 22; y++) {
        for (int x = 0; x < 22; x++) {
          List<double> blockPredictions = [];
          for (int c = 0; c < 17; c++) {
            blockPredictions.add(output[index++]);
          }

          // Process the predictions for this grid cell
          // Add detection if confidence threshold is met
          // This is where you'd implement your specific detection logic
          var processed = processGridCell(blockPredictions, x, y);
          if (processed != null) {
            results.add(processed);
          }
        }
      }
    } catch (e) {
      print('Error processing output: $e');
    }

    return results;
  }

  Map<String, dynamic>? processGridCell(List<double> cellValues, int x, int y) {
    // Implement your specific grid cell processing logic here
    // This is a placeholder - update according to your model's output format
    double confidence = cellValues[0]; // Assuming first value is confidence

    if (confidence > 0.5) {
      // Adjust threshold as needed
      return {
        "confidence": confidence,
        "x": x / 22.0, // Normalize coordinates
        "y": y / 22.0,
        // Add other relevant values based on your model's output format
      };
    }
    return null;
  }

  @override
  void dispose() {
    controller?.dispose();
    interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) {
      return Center(
          child: CircularProgressIndicator()); // Show a loading indicator
    }

    var tmp = MediaQuery.of(context).size;
    var screenH = math.max(tmp.height, tmp.width);
    var screenW = math.min(tmp.height, tmp.width);
    tmp = controller!.value.previewSize!;
    var previewH = math.max(tmp.height, tmp.width);
    var previewW = math.min(tmp.height, tmp.width);
    var screenRatio = screenH / screenW;
    var previewRatio = previewH / previewW;

    return OverflowBox(
      maxHeight:
          screenRatio > previewRatio ? screenH : screenW / previewW * previewH,
      maxWidth:
          screenRatio > previewRatio ? screenH / previewH * previewW : screenW,
      child: CameraPreview(controller!),
    );
  }
}
