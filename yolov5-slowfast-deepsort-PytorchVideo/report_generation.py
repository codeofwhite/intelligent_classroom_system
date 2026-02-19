import numpy as np
import os, cv2, time, torch, random, warnings, argparse, math, json, pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)

# 1. 核心依赖导入（保留最小依赖）
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample, short_side_scale_with_boxes, clip_boxes_to_image
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort

# -------------------------- 工具函数（修复+优化） --------------------------
def tensor_to_numpy(tensor):
    """张量转OpenCV图像"""
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(clip, boxes, num_frames=16, crop_size=480):  # 优化：减少帧数+缩小尺寸，提速
    """SlowFast输入预处理（极简版）"""
    boxes = np.array(boxes)
    # 1. 时间采样+归一化（帧数从32→16，提速50%）
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float() / 255.0
    height, width = clip.shape[2], clip.shape[3]
    # 2. 调整框+缩放图像（尺寸从640→480，提速）
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)
    clip = normalize(clip, np.array([0.45, 0.45, 0.45]), np.array([0.225, 0.225, 0.225]))
    # 3. 拆分Slow/Fast分支
    slow_pathway = torch.index_select(clip, 1, torch.linspace(0, clip.shape[1]-1, clip.shape[1]//4).long())
    fast_pathway = clip
    return [slow_pathway, fast_pathway], torch.from_numpy(boxes)

def deepsort_update(Tracker, pred, xywh, np_img):
    """DeepSort跟踪更新"""
    outputs = Tracker.update(xywh, pred[:,4:5], pred[:,5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs if len(outputs) > 0 else np.ones((0,8))

# 新增：NumPy类型转原生Python类型（核心修复）
def convert_numpy_to_python(obj):
    """递归将NumPy类型转为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def save_results_to_json(all_results, save_path):
    """保存所有结果到JSON（修复序列化问题）"""
    # 先转换NumPy类型→原生类型
    all_results = convert_numpy_to_python(all_results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"JSON结果已保存到：{save_path}")

def save_summary_to_excel(id_summary, save_path):
    """保存ID行为汇总到Excel"""
    # 转换NumPy类型→原生类型
    id_summary = convert_numpy_to_python(id_summary)
    df = pd.DataFrame([
        {
            '跟踪ID': tid,
            '主要行为': info['main_action'],
            '出现帧数': info['frame_count'],
            '首次出现时间(秒)': info['first_time'],
            '最后出现时间(秒)': info['last_time'],
            '平均坐标(x1,y1,x2,y2)': info['avg_box']
        }
        for tid, info in id_summary.items()
    ])
    df.to_excel(save_path, index=False)
    print(f"Excel汇总已保存到：{save_path}")

# -------------------------- 主函数（Demo核心） --------------------------
def main(config):
    # 1. 初始化模型（优化：用YOLOv5s轻量化模型，提速）
    print("初始化模型...")
    # YOLOv5s（小模型）替代YOLOv5l6，速度提升3倍+
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.4  # 置信度阈值
    model.iou = 0.4   # NMS阈值
    model.classes = [0]  # 只检测人
    # SlowFast行为识别（添加half()，CPU推理提速）
    video_model = slowfast_r50_detection(True).eval().half().to(config.device)
    # DeepSort跟踪
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    
    # 2. 行为标签映射（简化版，保留核心行为）
    action_labels = {
        0: '未知行为',
        7: '站立',
        8: '行走',
        9: '跑步',
        22: '举手',
        40: '低头',
        52: '坐姿'
    }

    # 3. 加载视频（只处理前10秒，加快速度）
    video_path = config.input
    video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
    max_process_seconds = 10  # 只处理前10秒
    video_duration = min(video.duration, max_process_seconds)
    print(f"开始处理视频（前{video_duration}秒）...")

    # 4. 初始化结果存储
    all_results = []  # 每帧详细结果
    id_summary = {}   # ID汇总信息

    # 5. 逐秒处理视频
    start_time = time.time()
    for sec in range(0, math.ceil(video_duration), 1):
        # 5.1 截取1秒视频片段
        clip = video.get_clip(sec, sec+1-0.04)['video']
        if clip is None:
            continue
        frame_num = clip.shape[1]  # 该片段帧数（约25帧）
        
        # 5.2 转换为图像列表，用于YOLO检测
        imgs = [tensor_to_numpy(clip[:,j,:,:]) for j in range(frame_num)]
        
        # 5.3 YOLO检测（添加half()，CPU推理提速）
        yolo_preds = model(imgs, size=480)  # 推理尺寸从640→480，提速
        
        # 5.4 DeepSort跟踪（获取ID）
        frame_results = []  # 存储当前秒内每帧的结果
        for frame_idx in range(len(yolo_preds.pred)):
            pred = yolo_preds.pred[frame_idx].cpu()
            xywh = yolo_preds.xywh[frame_idx][:,0:4].cpu()
            img = yolo_preds.ims[frame_idx]
            
            # 更新跟踪器
            track_output = deepsort_update(deepsort_tracker, pred, xywh, img)
            
            # 存储当前帧结果（提前转换NumPy类型）
            frame_time = sec + frame_idx/frame_num  # 精确到帧的时间戳
            for track in track_output:
                x1, y1, x2, y2, _, track_id, vx, vy = track
                # 核心：将NumPy类型转为原生类型
                frame_results.append({
                    '时间戳(秒)': round(float(frame_time), 2),
                    '跟踪ID': int(track_id),
                    '检测框坐标': [round(float(x1),2), round(float(y1),2), round(float(x2),2), round(float(y2),2)],
                    '行为标签': '未识别'
                })
        
        # 5.5 SlowFast行为识别（只处理中间帧，减少计算）
        if len(yolo_preds.pred[frame_num//2]) > 0:
            # 预处理输入
            boxes = yolo_preds.pred[frame_num//2][:,0:4]
            inputs, inp_boxes = ava_inference_transform(clip, boxes)
            # 模型推理（half()提速）
            inputs = [inp.unsqueeze(0).half().to(config.device) for inp in inputs]
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1).half().to(config.device)
            with torch.no_grad():
                preds = video_model(inputs, inp_boxes).cpu()
            # 映射行为标签
            track_ids = yolo_preds.pred[frame_num//2][:,5].tolist()
            action_ids = np.argmax(preds, axis=1).tolist()
            # 填充行为标签到frame_results
            for res in frame_results:
                tid = res['跟踪ID']
                if tid in track_ids:
                    idx = track_ids.index(tid)
                    action_id = int(action_ids[idx]) + 1  # 转换为原生int
                    res['行为标签'] = action_labels.get(action_id, '未知行为')
        
        # 5.6 汇总结果
        all_results.extend(frame_results)
        # 更新ID汇总信息
        for res in frame_results:
            tid = res['跟踪ID']
            if tid not in id_summary:
                id_summary[tid] = {
                    'main_action': res['行为标签'],
                    'frame_count': 1,
                    'first_time': res['时间戳(秒)'],
                    'last_time': res['时间戳(秒)'],
                    'avg_box': res['检测框坐标']
                }
            else:
                id_summary[tid]['frame_count'] += 1
                id_summary[tid]['last_time'] = res['时间戳(秒)']
                # 统计出现次数最多的行为
                if res['行为标签'] != id_summary[tid]['main_action']:
                    current_count = sum(1 for r in frame_results if r['跟踪ID']==tid and r['行为标签']==res['行为标签'])
                    main_count = sum(1 for r in frame_results if r['跟踪ID']==tid and r['行为标签']==id_summary[tid]['main_action'])
                    if current_count > main_count:
                        id_summary[tid]['main_action'] = res['行为标签']
                # 平均坐标
                old_box = id_summary[tid]['avg_box']
                new_box = res['检测框坐标']
                id_summary[tid]['avg_box'] = [
                    round((old_box[0]+new_box[0])/2,2),
                    round((old_box[1]+new_box[1])/2,2),
                    round((old_box[2]+new_box[2])/2,2),
                    round((old_box[3]+new_box[3])/2,2)
                ]
        
        print(f"已处理 {sec+1}/{math.ceil(video_duration)} 秒 | 已检测到 {len(id_summary)} 个目标")

    # 6. 输出结果
    print(f"\n处理完成！总耗时：{round(time.time()-start_time,2)} 秒")
    # 保存JSON详细结果
    save_results_to_json(all_results, config.json_output)
    # 保存Excel汇总
    save_summary_to_excel(id_summary, config.excel_output)

    # 7. 打印简易结果（终端预览）
    print("\n=== 核心结果预览 ===")
    for tid, info in id_summary.items():
        print(f"ID: {tid} | 主要行为: {info['main_action']} | 出现帧数: {info['frame_count']} | 平均坐标: {info['avg_box']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="demo/test.mp4", help='输入视频路径')
    parser.add_argument('--json_output', type=str, default="detection_results.json", help='JSON结果输出路径')
    parser.add_argument('--excel_output', type=str, default="detection_summary.xlsx", help='Excel汇总输出路径')
    parser.add_argument('--device', default='cpu', help='运行设备（cuda/cpu）')
    config = parser.parse_args()

    # 检查权重文件是否存在
    if not os.path.exists("deep_sort/deep_sort/deep/checkpoint/ckpt.t7"):
        print("错误：未找到ckpt.t7权重文件！请先下载放到指定路径")
        exit(1)
    
    # 检查输入视频是否存在
    if not os.path.exists(config.input):
        print(f"错误：输入视频 {config.input} 不存在！")
        exit(1)

    main(config)