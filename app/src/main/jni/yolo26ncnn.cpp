#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

static Yolo* g_yolo = 0;
static ncnn::Mutex lock;

// YOLO26n 配置
static const int YOLO26_TARGET_SIZE = 640;
static const float YOLO26_MEAN_VALS[3] = {0.f, 0.f, 0.f};
static const float YOLO26_NORM_VALS[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "JNI_OnUnload");

    ncnn::MutexLockGuard g(lock);
    delete g_yolo;
    g_yolo = 0;
}

JNIEXPORT jboolean JNICALL Java_com_example_yolo26ncnn_Yolo26Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint useGpu) {
    if (modelid < 0 || modelid > 0) {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "loadModel %p", mgr);

    const char* modeltype = "yolo26n";
    bool use_gpu = (useGpu == 1);

    {
        ncnn::MutexLockGuard g(lock);

        // Check GPU availability
        if (use_gpu && ncnn::get_gpu_count() == 0) {
            __android_log_print(ANDROID_LOG_WARN, "Yolo26Ncnn", "GPU not available, falling back to CPU");
            use_gpu = false;
        }

        if (!g_yolo) {
            g_yolo = new Yolo;
        }

        const char* device_name = use_gpu ? "GPU (FP32)" : "CPU";
        __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "Loading model: %s on %s", modeltype, device_name);
        g_yolo->load(mgr, modeltype, YOLO26_TARGET_SIZE, YOLO26_MEAN_VALS, YOLO26_NORM_VALS, use_gpu);
        __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "Model loaded successfully");
    }

    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL Java_com_example_yolo26ncnn_Yolo26Ncnn_detect(JNIEnv* env, jobject thiz, jobject bitmap) {
    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // Lock bitmap pixels
    void* indata;
    AndroidBitmap_lockPixels(env, bitmap, &indata);

    int width  = info.width;
    int height = info.height;

    // 使用 ncnn::Mat::from_pixels_resize 直接完成：
    // RGBA→BGR 转换 + letterbox resize + padding
    // 无需 OpenCV 的 cvtColor

    // letterbox scale to 640x640
    const int dst_size = YOLO26_TARGET_SIZE;
    float scale = std::min(dst_size / (float)width, dst_size / (float)height);
    int new_w = (int)std::round(width * scale);
    int new_h = (int)std::round(height * scale);

    int wpad = dst_size - new_w;
    int hpad = dst_size - new_h;
    int pad_left = wpad / 2;
    int pad_top  = hpad / 2;

    // RGBA → BGR + resize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            (unsigned char*)indata,
            ncnn::Mat::PIXEL_RGBA2BGR,
            width, height,
            new_w, new_h
    );

    // padding 到 640x640
    ncnn::Mat in_pad;
    ncnn::copy_make_border(
            in, in_pad,
            pad_top, hpad - pad_top,
            pad_left, wpad - pad_left,
            ncnn::BORDER_CONSTANT,
            114.f
    );

    AndroidBitmap_unlockPixels(env, bitmap);

    // Detection
    std::vector<Object> objects;
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolo) {
            g_yolo->detect(in_pad, objects);
        }
    }

    // 将检测结果坐标映射回原图尺寸（撤销 letterbox padding 和 scale）
    for (size_t i = 0; i < objects.size(); i++) {
        float x0 = (objects[i].rect.x - (float)pad_left) / scale;
        float y0 = (objects[i].rect.y - (float)pad_top)  / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width  - (float)pad_left) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (float)pad_top)  / scale;

        x0 = std::max(std::min(x0, (float)(width  - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width  - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x      = x0;
        objects[i].rect.y      = y0;
        objects[i].rect.width  = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // Create result array
    jclass objCls = env->FindClass("com/example/yolo26ncnn/Yolo26Ncnn$Obj");
    jmethodID objInit = env->GetMethodID(objCls, "<init>", "(Lcom/example/yolo26ncnn/Yolo26Ncnn;)V");
    jfieldID xId = env->GetFieldID(objCls, "x", "F");
    jfieldID yId = env->GetFieldID(objCls, "y", "F");
    jfieldID wId = env->GetFieldID(objCls, "w", "F");
    jfieldID hId = env->GetFieldID(objCls, "h", "F");
    jfieldID labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    jfieldID probId = env->GetFieldID(objCls, "prob", "F");

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i = 0; i < objects.size(); i++) {
        jobject jObj = env->NewObject(objCls, objInit, thiz);

        env->SetFloatField(jObj, xId, objects[i].rect.x);
        env->SetFloatField(jObj, yId, objects[i].rect.y);
        env->SetFloatField(jObj, wId, objects[i].rect.width);
        env->SetFloatField(jObj, hId, objects[i].rect.height);

        int label = objects[i].label;
        const char* label_name = (label >= 0 && label < 80) ? class_names[label] : "unknown";
        env->SetObjectField(jObj, labelId, env->NewStringUTF(label_name));
        env->SetFloatField(jObj, probId, objects[i].prob);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "Yolo26Ncnn", "%.2fms detect", elasped);

    return jObjArray;
}

}
