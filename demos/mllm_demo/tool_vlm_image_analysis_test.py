import dashscope
dashscope.api_key = "sk-06abd7a7eb514b3ebd611412f0dc3531"

image_path = "key_frames/global_frame_30_distract_16.jpg"

print("测试 通义 VLM 官方正确格式...")

try:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张课堂图片"},
                # 🔥 只有这行是关键修复！！！
                {"type": "image", "image": f"file://{image_path}"}
            ]
        }
    ]

    resp = dashscope.MultiModalConversation.call(
        model="qwen-vl-max",
        messages=messages
    )

    print("✅ VLM 调用成功！")
    print("返回结果：", resp.output.choices[0].message.content)

except Exception as e:
    print("❌ 错误：", e)