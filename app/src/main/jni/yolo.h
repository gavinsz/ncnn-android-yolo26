#ifndef YOLO_H
#define YOLO_H

#include <net.h>

struct Rect {
    float x;
    float y;
    float width;
    float height;

    float area() const { return width * height; }
};

struct Object {
    Rect rect;
    int label;
    float prob;
};

class Yolo {
public:
    Yolo();
    ~Yolo();

    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    // 输入: ncnn::Mat, pixel_type 传 ncnn::Mat::PIXEL_RGBA2BGR
    // （camera 送来 RGBA，模型期望 BGR，直接在 ncnn 内部转换）
    int detect(ncnn::Mat& in, std::vector<Object>& objects, float prob_threshold = 0.25f, float nms_threshold = 0.45f);

private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

// COCO 80类名称
static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

#endif // YOLO_H
