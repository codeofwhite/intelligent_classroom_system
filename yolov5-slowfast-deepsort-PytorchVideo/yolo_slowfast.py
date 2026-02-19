import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
warnings.filterwarnings("ignore",category=UserWarning) # 忽略无关警告，避免输出杂乱

# PytorchVideo 相关的视频变换函数（核心用于SlowFast的输入预处理）
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,  # 均匀时间采样（从视频片段中选固定帧数）
    short_side_scale_with_boxes,  # 缩放视频短边，同时调整检测框坐标
    clip_boxes_to_image,)         # 裁剪检测框，确保框在图像范围内
from torchvision.transforms._functional_video import normalize  # 视频帧归一化
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths    # 加载AVA动作标签映射
from pytorchvideo.models.hub import slowfast_r50_detection     # 加载SlowFast动作检测模型
from deep_sort.deep_sort import DeepSort  # 导入DeepSort跟踪器

def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))  # (C,T,H,W) → (T,H,W,C)，适配OpenCV格式
    return img

def ava_inference_transform(clip, boxes,
    num_frames = 32,  # SlowFast的输入帧数（slow分支8帧+fast分支32帧）
    crop_size = 640,  # 图像裁剪尺寸
    data_mean = [0.45, 0.45, 0.45],  # 归一化均值
    data_std = [0.225, 0.225, 0.225], # 归一化方差
    slow_fast_alpha = 4, # Slow/Fast分支的采样比例（fast是slow的4倍）
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()  # 保存原始检测框（后续还原用）
    # 1. 时间维度均匀采样：从视频片段中选32帧
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float() / 255.0  # 像素值归一化到0-1
    height, width = clip.shape[2], clip.shape[3]
    # 2. 确保检测框在图像范围内
    boxes = clip_boxes_to_image(boxes, height, width)
    # 3. 缩放图像短边，同时调整检测框坐标（保持比例）
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    # 4. 图像归一化（减均值、除方差），适配模型输入要求
    clip = normalize(clip, np.array(data_mean), np.array(data_std))
    # 5. 再次裁剪框，确保缩放后框仍在图像内
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    # 6. 构建SlowFast的双分支输入（slow分支采样稀疏，fast分支采样密集）
    if slow_fast_alpha is not None:
        fast_pathway = clip  # fast分支：全部32帧
        # slow分支：从32帧中均匀选8帧（32//4=8）
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]  # SlowFast模型需要双分支输入
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
    # x是检测框坐标 [x1,y1,x2,y2]，img是图像数组
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 1. 绘制矩形框
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    # 2. 计算文本尺寸，绘制文本背景框
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    # 3. 绘制文本（ID+类别+动作）
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker,pred,xywh,np_img):
    # Tracker：DeepSort实例；pred：Yolov5检测结果；xywh：检测框的xywh格式；np_img：图像数组
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,color_map,output_video):
    # yolo_preds：Yolov5+DeepSort的结果；id_to_ava_labels：ID到动作标签的映射；output_video：视频写入器
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  # 转换色彩空间
        if pred.shape[0]:  # 如果有检测结果
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                # 只处理人（cls=0），其他类别跳过
                if int(cls) != 0:
                    ava_label = ''
                # 如果该ID有动作标签，取标签；否则标为Unknown
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                # 拼接文本：ID + 类别名 + 动作标签
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]  # 按类别取颜色
                im = plot_one_box(box,im,color,text)  # 绘制框和文本
        output_video.write(im.astype(np.uint8))  # 写入视频帧

def main(config):
    # 1. 加载Yolov5l6模型（large版本，检测精度更高）
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
    model.conf = config.conf  # 检测置信度阈值（0.4）
    model.iou = config.iou    # NMS的IOU阈值（0.4）
    model.max_det = 200       # 单帧最大检测目标数
    if config.classes:
        model.classes = config.classes  # 过滤检测类别（默认检测人）
    device = config.device    # 运行设备（cuda/cpu）
    imsize = config.imsize    # 推理尺寸（640）
    
    # 2. 加载SlowFast动作检测模型（预训练，用于动作识别）
    video_model = slowfast_r50_detection(True).eval().to(device)
    
    # 3. 初始化DeepSort跟踪器（加载预训练权重）
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    
    # 4. 加载AVA动作标签映射（将模型输出的数字ID转为动作名称，如“走路”“跑步”）
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    
    # 5. 生成COCO类别颜色映射（每个类别对应随机颜色，用于绘制框）
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    # 1. 配置输出视频路径和参数
    vide_save_path = config.output
    # 先打开视频获取宽高，再释放（仅用于获取尺寸）
    video=cv2.VideoCapture(config.input)
    width,height = int(video.get(3)),int(video.get(4))
    video.release()
    # 初始化视频写入器（MP4格式，25帧/秒，尺寸与原视频一致）
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
    print("processing...")
    
    # 2. 用PytorchVideo加载视频（支持按时间段截取片段，更适合视频理解）
    video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(config.input)
    a=time.time()  # 记录开始时间，用于计算总耗时
    
    # 按1秒为单位遍历视频（步长1秒，每次处理1秒的视频片段）
    for i in range(0,math.ceil(video.duration),1):
        # 1. 截取1秒的视频片段（i到i+1秒，减0.04是避免帧重复）
        video_clips=video.get_clip(i, i+1-0.04)
        video_clips=video_clips['video']  # 提取视频张量 (C, T, H, W)
        if video_clips is None:  # 无有效片段则跳过
            continue
        img_num=video_clips.shape[1]  # 该片段的帧数（约25帧，对应1秒）
        imgs=[]
        # 2. 将张量转为OpenCV能处理的图像数组，存入列表
        for j in range(img_num):
            imgs.append(tensor_to_numpy(video_clips[:,j,:,:]))
        
        # 3. Yolov5检测：对该片段的所有帧做目标检测
        yolo_preds=model(imgs, size=imsize)
        yolo_preds.files=[f"img_{i*25+k}.jpg" for k in range(img_num)]  # 给帧命名（无实际作用，仅标记）

        print(i,video_clips.shape,img_num)  # 打印进度：当前秒数、张量形状、帧数
        deepsort_outputs=[]
        # 4. DeepSort跟踪：对每一帧的检测结果更新跟踪ID
        for j in range(len(yolo_preds.pred)):
            temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.ims[j])
            if len(temp)==0:  # 无跟踪结果则补空数组
                temp=np.ones((0,8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred=deepsort_outputs  # 将跟踪结果替换原检测结果（新增了ID）

        # 5. SlowFast动作识别：仅对片段中间帧的检测结果做动作识别（减少计算量）
        id_to_ava_labels={}  # 存储“跟踪ID→动作标签”的映射
        if yolo_preds.pred[img_num//2].shape[0]:  # 中间帧有检测结果才执行
            # 5.1 预处理视频片段和检测框，适配SlowFast输入
            inputs,inp_boxes,_=ava_inference_transform(video_clips,yolo_preds.pred[img_num//2][:,0:4],crop_size=imsize)
            # 5.2 给检测框加batch维度（模型输入要求）
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
            # 5.3 调整输入格式，移到指定设备（cuda/cpu）
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            # 5.4 推理：无梯度计算（加速），获取动作预测结果
            with torch.no_grad():
                slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                slowfaster_preds = slowfaster_preds.cpu()
            # 5.5 映射动作标签：将预测的数字ID转为动作名称
            for tid,avalabel in zip(yolo_preds.pred[img_num//2][:,5].tolist(),np.argmax(slowfaster_preds,axis=1).tolist()):
                id_to_ava_labels[tid]=ava_labelnames[avalabel+1]
        
        # 6. 保存当前片段的所有帧到输出视频（绘制框和标签）
        save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,coco_color_map,outputvideo)    
    
    outputvideo.release()  # 释放视频写入器
    print("total time:",time.time()-a)  # 打印总耗时
    
if __name__=="__main__":
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="demo/test.mp4", help='输入视频路径')
    parser.add_argument('--output', type=str, default="output.mp4", help='输出视频路径')
    parser.add_argument('--imsize', type=int, default=640, help='推理尺寸')
    parser.add_argument('--conf', type=float, default=0.4, help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.4, help='NMS的IOU阈值')
    parser.add_argument('--device', default='cpu', help='运行设备（cuda/cpu）')
    parser.add_argument('--classes', nargs='+', type=int, help='过滤检测类别（如--classes 0 只检测人）')
    config = parser.parse_args()  # 解析参数
    
    print(config)
    main(config)  # 执行主函数
