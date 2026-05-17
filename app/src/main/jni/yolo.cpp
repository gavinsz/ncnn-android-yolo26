// YOLO26 ncnn implementation
// out0: dims=2, w=8400, h=84  => [84 rows, 8400 cols]
// row 0..3 : cx, cy, w, h (decoded in 640x640 coords)
// row 4..83: 80 class probs (sigmoid already in graph)

#include "yolo.h"

#include <cpu.h>
#include <layer.h>

#include <android/log.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <cmath>

#define TAG "YOLO26"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

// 计算两个矩形交集面积（替代 cv::Rect_<float> 的 & 操作）
static inline float intersection_area(const Object& a, const Object& b)
{
    float x1 = std::max(a.rect.x, b.rect.x);
    float y1 = std::max(a.rect.y, b.rect.y);
    float x2 = std::min(a.rect.x + a.rect.width,  b.rect.x + b.rect.width);
    float y2 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
    if (x2 < x1 || y2 < y1) return 0.f;
    return (x2 - x1) * (y2 - y1);
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, (int)objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = (int)objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = objects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = union_area > 0.f ? (inter_area / union_area) : 0.f;

            if (iou > nms_threshold)
            {
                keep = 0;
                break;
            }
        }

        if (keep) picked.push_back(i);
    }
}

static void generate_proposals_yolo26(const ncnn::Mat& pred,
                                      float prob_threshold,
                                      std::vector<Object>& objects,
                                      float* out_global_max = nullptr)
{
    objects.clear();

    if (pred.dims != 2)
    {
        LOGD("generate_proposals: unexpected pred.dims=%d (expected 2)", pred.dims);
        if (out_global_max) *out_global_max = 0.f;
        return;
    }

    const int num_proposals = pred.w;        // 8400
    const int num_feat      = pred.h;        // 84
    const int num_class     = num_feat - 4;  // 80

    if (num_feat != 84)
    {
        LOGD("generate_proposals: unexpected pred.h=%d (expected 84=4+80)", num_feat);
        if (out_global_max) *out_global_max = 0.f;
        return;
    }

    const float* ptr_cx = pred.row(0);
    const float* ptr_cy = pred.row(1);
    const float* ptr_w  = pred.row(2);
    const float* ptr_h  = pred.row(3);

    float global_max = 0.f;

    for (int i = 0; i < num_proposals; i++)
    {
        int label = -1;
        float score = 0.f;

        for (int k = 0; k < num_class; k++)
        {
            const float* row_cls = pred.row(4 + k);
            float s = row_cls[i]; // already sigmoid
            if (s > score)
            {
                score = s;
                label = k;
            }
        }

        if (score > global_max) global_max = score;
        if (score < prob_threshold) continue;

        float cx = ptr_cx[i];
        float cy = ptr_cy[i];
        float bw = ptr_w[i];
        float bh = ptr_h[i];

        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width  = bw;
        obj.rect.height = bh;
        obj.label = label;
        obj.prob  = score;
        objects.push_back(obj);
    }

    if (out_global_max) *out_global_max = global_max;
}

Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

Yolo::~Yolo()
{
    yolo.clear();
}

int Yolo::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
    if (use_gpu)
    {
        yolo.opt.use_fp16_packed = false;
        yolo.opt.use_fp16_storage = false;
        yolo.opt.use_fp16_arithmetic = false;
    }
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    yolo.load_param(parampath);
    yolo.load_model(modelpath);

    target_size = _target_size;

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int Yolo::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
    if (use_gpu)
    {
        yolo.opt.use_fp16_packed = false;
        yolo.opt.use_fp16_storage = false;
        yolo.opt.use_fp16_arithmetic = false;
    }
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);

    target_size = _target_size;

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

// detect: in 是已经完成 RGBA→BGR 转换 + letterbox resize + padding 的 ncnn::Mat
int Yolo::detect(ncnn::Mat& in_pad, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    objects.clear();

    const int img_w = in_pad.w;
    const int img_h = in_pad.h;

    // 归一化
    const float mean_vals_ultra[3] = {0.f, 0.f, 0.f};
    const float norm_vals_ultra[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(mean_vals_ultra, norm_vals_ultra);

    ncnn::Extractor ex = yolo.create_extractor();
    ex.set_light_mode(true);

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    LOGD("YOLO26 output: dims=%d, w=%d (proposals), h=%d (features), c=%d",
         out.dims, out.w, out.h, out.c);

    std::vector<Object> proposals;
    generate_proposals_yolo26(out, prob_threshold, proposals, nullptr);

    if (proposals.empty())
        return 0;

    // sort by score desc
    qsort_descent_inplace(proposals);

    // NMS (set nms_threshold<=0 to disable)
    std::vector<int> picked;
    if (nms_threshold > 0.f)
        nms_sorted_bboxes(proposals, picked, nms_threshold);
    else
    {
        picked.resize(proposals.size());
        for (int i = 0; i < (int)proposals.size(); i++) picked[i] = i;
    }

    LOGD("after NMS: %zu", picked.size());

    int count = (int)picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }

    // sort by area desc (optional)
    struct {
        bool operator()(const Object& a, const Object& b) const {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;

    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}
