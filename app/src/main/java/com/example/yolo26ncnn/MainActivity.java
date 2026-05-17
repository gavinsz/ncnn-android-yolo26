package com.example.yolo26ncnn;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Yolo26Ncnn";
    private static final int REQUEST_PERMISSION = 100;
    private static final String[] REQUIRED_PERMISSIONS;

    static {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // API 33+：相机 + 细粒度媒体权限
            REQUIRED_PERMISSIONS = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_MEDIA_IMAGES
            };
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // API 23-32：相机 + 外部存储读取
            REQUIRED_PERMISSIONS = new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.READ_EXTERNAL_STORAGE
            };
        } else {
            // API 21-22：权限在安装时自动授予，只列相机即可
            REQUIRED_PERMISSIONS = new String[]{
                    Manifest.permission.CAMERA
            };
        }
    }

    private Yolo26Ncnn yolo26Ncnn = new Yolo26Ncnn();
    private PreviewView previewView;
    private OverlayView overlayView;
    private TextView tvResult;
    private Button btnSwitchCamera;
    private Button btnStartStop;
    private Spinner spinnerModel;
    private Spinner spinnerDevice;

    private int currentModel = 0;
    private int currentDevice = 0; // 0=CPU, 1=GPU
    private boolean isDetecting = false;
    private boolean isFrontCamera = false;

    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;

    private long lastDetectTime = 0;
    private static final long DETECT_INTERVAL = 100; // 检测间隔 100ms

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);
        tvResult = findViewById(R.id.tvResult);
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera);
        btnStartStop = findViewById(R.id.btnStartStop);
        spinnerModel = findViewById(R.id.spinnerModel);
        spinnerDevice = findViewById(R.id.spinnerDevice);

        // Set PreviewView to fill the container (crop to fit)
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Model selection
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                currentModel = position;
                reloadModel();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // Device selection (CPU/GPU)
        spinnerDevice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                currentDevice = position;
                reloadModel();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // Switch camera button
        btnSwitchCamera.setOnClickListener(v -> {
            isFrontCamera = !isFrontCamera;
            overlayView.setFrontCamera(isFrontCamera);
            startCamera();
        });

        // Start/Stop detection button
        btnStartStop.setOnClickListener(v -> {
            isDetecting = !isDetecting;
            btnStartStop.setText(isDetecting ? "Stop Detect" : "Start Detect");
            if (!isDetecting) {
                overlayView.clearResults();
                tvResult.setText("Detection stopped");
            }
        });

        // 检查权限
        if (allPermissionsGranted()) {
            reloadModel();
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_PERMISSION);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION) {
            if (allPermissionsGranted()) {
                reloadModel();
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    private void reloadModel() {
        boolean ret = yolo26Ncnn.loadModel(getAssets(), currentModel, currentDevice);
        if (!ret) {
            Log.e(TAG, "Failed to load model");
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
        } else {
            String device = currentDevice == 0 ? "CPU" : "GPU (FP32)";
            Log.d(TAG, "Model loaded: yolo26n on " + device);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Camera initialization failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) return;

        // 解绑之前的用例
        cameraProvider.unbindAll();

        // 选择摄像头
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(isFrontCamera ?
                        CameraSelector.LENS_FACING_FRONT :
                        CameraSelector.LENS_FACING_BACK)
                .build();

        // 分辨率选择器（替代已废弃的 setTargetResolution）
        ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                .setResolutionStrategy(new ResolutionStrategy(
                        new Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build();

        // 预览
        Preview preview = new Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // 图像分析
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);

        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
            runOnUiThread(() -> tvResult.setText("Camera ready. Tap \"Start Detect\" to begin."));
        } catch (Exception e) {
            Log.e(TAG, "Use case binding failed", e);
        }
    }

    private void analyzeImage(ImageProxy image) {
        if (!isDetecting) {
            image.close();
            return;
        }

        // 控制检测频率
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastDetectTime < DETECT_INTERVAL) {
            image.close();
            return;
        }
        lastDetectTime = currentTime;

        try {
            // 将 ImageProxy 转换为 Bitmap
            Bitmap bitmap = imageProxyToBitmap(image);
            if (bitmap == null) {
                image.close();
                return;
            }

            // 设置预览尺寸
            int previewWidth = bitmap.getWidth();
            int previewHeight = bitmap.getHeight();

            // Run detection
            long startTime = System.currentTimeMillis();
            Yolo26Ncnn.Obj[] objects = yolo26Ncnn.detect(bitmap);
            long inferenceTime = System.currentTimeMillis() - startTime;

            // Update UI
            final int objectCount = objects != null ? objects.length : 0;
            final long fps = 1000 / Math.max(inferenceTime, 1);

            runOnUiThread(() -> {
                overlayView.setPreviewSize(previewWidth, previewHeight);
                overlayView.setResults(objects);

                if (objectCount > 0) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(String.format("Detected %d objects | %dms | %d FPS\n", objectCount, inferenceTime, fps));
                    for (int i = 0; i < Math.min(objectCount, 3); i++) {
                        if (objects[i].label != null) {
                            sb.append(String.format("%s: %.1f%% ", objects[i].label, objects[i].prob * 100));
                        }
                    }
                    if (objectCount > 3) {
                        sb.append("...");
                    }
                    tvResult.setText(sb.toString());
                } else {
                    tvResult.setText(String.format("No objects detected | %dms | %d FPS", inferenceTime, fps));
                }
            });

            bitmap.recycle();

        } catch (Exception e) {
            Log.e(TAG, "Detection failed", e);
        } finally {
            image.close();
        }
    }

    /**
     * 将 ImageProxy (YUV_420_888) 正确转换为 Bitmap
     * 关键：正确处理 rowStride 和 pixelStride
     */
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        try {
            ImageProxy.PlaneProxy[] planes = image.getPlanes();
            int width = image.getWidth();
            int height = image.getHeight();

            // Y 平面
            ByteBuffer yBuffer = planes[0].getBuffer();
            int yRowStride = planes[0].getRowStride();
            int yPixelStride = planes[0].getPixelStride();

            // U 平面
            ByteBuffer uBuffer = planes[1].getBuffer();
            int uvRowStride = planes[1].getRowStride();
            int uvPixelStride = planes[1].getPixelStride();

            // V 平面
            ByteBuffer vBuffer = planes[2].getBuffer();

            // NV21 格式: Y 平面 + VU 交错平面
            // 总大小 = width * height * 1.5
            int nv21Size = width * height + (width * height / 2);
            byte[] nv21 = new byte[nv21Size];

            // 复制 Y 平面 (考虑 rowStride)
            int yPos = 0;
            if (yRowStride == width && yPixelStride == 1) {
                // 快速路径：直接复制
                yBuffer.get(nv21, 0, width * height);
                yPos = width * height;
            } else {
                // 慢速路径：逐行复制
                for (int row = 0; row < height; row++) {
                    yBuffer.position(row * yRowStride);
                    for (int col = 0; col < width; col++) {
                        nv21[yPos++] = yBuffer.get(row * yRowStride + col * yPixelStride);
                    }
                }
                yBuffer.rewind();
            }

            // 复制 VU 交错平面
            // NV21 要求 V 在前，U 在后，交错存储
            int uvHeight = height / 2;
            int uvWidth = width / 2;
            int uvPos = width * height;

            if (uvPixelStride == 2 && uvRowStride == width) {
                // 快速路径：UV 已经是交错的 (常见情况)
                // planes[2] (V) 已经是 VUVU... 格式
                vBuffer.position(0);
                int uvSize = Math.min(vBuffer.remaining(), width * height / 2);
                vBuffer.get(nv21, uvPos, uvSize);
            } else if (uvPixelStride == 1) {
                // 慢速路径：U 和 V 是分开的平面，需要手动交错
                for (int row = 0; row < uvHeight; row++) {
                    for (int col = 0; col < uvWidth; col++) {
                        int uvIndex = row * uvRowStride + col * uvPixelStride;
                        // NV21: V 先，U 后
                        nv21[uvPos++] = vBuffer.get(uvIndex);  // V
                        nv21[uvPos++] = uBuffer.get(uvIndex);  // U
                    }
                }
            } else {
                // 通用路径
                for (int row = 0; row < uvHeight; row++) {
                    for (int col = 0; col < uvWidth; col++) {
                        int uvIndex = row * uvRowStride + col * uvPixelStride;
                        nv21[uvPos++] = vBuffer.get(uvIndex);  // V
                        nv21[uvPos++] = uBuffer.get(uvIndex);  // U
                    }
                }
            }

            // 使用 YuvImage 转换为 JPEG，然后解码为 Bitmap
            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, width, height), 95, out);
            byte[] imageBytes = out.toByteArray();

            Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

            // 旋转图像
            Matrix matrix = new Matrix();
            int rotationDegrees = image.getImageInfo().getRotationDegrees();
            if (rotationDegrees != 0) {
                matrix.postRotate(rotationDegrees);
            }

            // 如果是前置摄像头，需要镜像翻转
            if (isFrontCamera) {
                matrix.postScale(-1, 1);
            }

            if (rotationDegrees != 0 || isFrontCamera) {
                Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                if (rotatedBitmap != bitmap) {
                    bitmap.recycle();
                }
                return rotatedBitmap;
            }

            return bitmap;

        } catch (Exception e) {
            Log.e(TAG, "Image conversion failed", e);
            return null;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }
}
