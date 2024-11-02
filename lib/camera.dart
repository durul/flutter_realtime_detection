import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
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
    // Example processing - you'll need to implement the actual processing logic
    // based on your model type
    try {
      final List<dynamic> results = []; // Add your model inference results here

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
