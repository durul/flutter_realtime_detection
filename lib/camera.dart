import 'dart:math' as math;

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

  Future<void> initializeCamera() async {
    if (widget.cameras.isEmpty) {
      print('No camera is found');
      return;
    }
    // Initialize the appropriate model
    try {
      print('Loading model: ${widget.model}');
      if (widget.model == mobilenet) {
        interpreter =
            await Interpreter.fromAsset('assets/mobilenet_v1_1.0_224.tflite');
      } else if (widget.model == posenet) {
        interpreter = await Interpreter.fromAsset(
            'assets/posenet_mv1_075_float_from_checkpoints.tflite');
      } else {
        interpreter = await Interpreter.fromAsset(widget.model == ssd
            ? 'assets/ssd_mobilenet.tflite'
            : 'assets/yolov2_tiny.tflite');
      }
      print('Model loaded successfully: ${widget.model}');
    } catch (e) {
      print('Error loading model: $e');
      return;
    }

    controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      print('Initializing camera...');
      await controller!.initialize();
      if (!mounted) return;

      setState(() {});
      print('Camera initialized successfully.');

      controller!.startImageStream((CameraImage img) {
        if (!isDetecting) {
          isDetecting = true;
          print('Processing image...');

          processImage(img).then((_) {
            isDetecting = false;
            print('Image processed.');
          }).catchError((e) {
            print('Error processing image: $e');
            isDetecting = false;
          });
        }
      });
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  Future<void> processImage(CameraImage image) async {
    try {
      // Convert the image to the format expected by the model
      // This is a placeholder for actual preprocessing logic
      var input = preprocessImage(image);

      // Define the output buffer
      var output =
          List.filled(1 * 10, 0).reshape([1, 10]); // Adjust size as needed

      // Run inference
      interpreter.run(input, output);

      // Process the output to extract meaningful results
      final List<dynamic> results = processOutput(output);

      // Update recognitions
      if (results.isNotEmpty) {
        widget.setRecognitions(
          results,
          image.height,
          image.width,
        );
      }
    } catch (e) {
      print('Error during image processing: $e');
    }
  }

// Placeholder for image preprocessing

  List<double> preprocessImage(CameraImage image) {
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

    // Resize the image to 224x224 (standard input size for many models)
    final img.Image resizedImage = img.copyResize(rgbImage,
        width: 224, height: 224, interpolation: img.Interpolation.linear);

    // Convert to normalized float array
    List<double> normalized = [];
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resizedImage.getPixel(x, y);
        // In image package v4.3.0, we need to use pixel.r, pixel.g, pixel.b
        normalized.add(pixel.r.toDouble() / 255.0);
        normalized.add(pixel.g.toDouble() / 255.0);
        normalized.add(pixel.b.toDouble() / 255.0);
      }
    }

    return normalized;
  }

// Placeholder for processing model output
  List<dynamic> processOutput(List output) {
    // Assuming the output is a list of predictions
    List<dynamic> results = [];

    for (var prediction in output) {
      // Process each prediction
      // This is a placeholder logic, adjust based on your model's output format
      if (prediction['confidence'] > 0.5) {
        results.add({
          "label": prediction['label'],
          "confidence": prediction['confidence'],
          "rect": prediction['rect'],
        });
      }
    }

    return results;
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
