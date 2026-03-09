# Program title: Storytelling App

# import part
import streamlit as st
from transformers import pipeline
import os
import tempfile

# img2text
def img2text(image_path):
    """
    使用BLIP模型将图像转换为文字描述
    """
    try:
        # 加载模型（每次调用都会重新加载）
        image_to_text = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
        result = image_to_text(image_path)
        return result[0]["generated_text"]
    except Exception as e:
        st.error(f"图像描述生成失败：{e}")
        return None

# ============================================================
# 故事生成功能（控制字数不超过100词）
# ============================================================
def text2story(caption, max_words=100):
    """
    基于图像描述生成儿童故事，并控制字数
    """
    try:
        # 加载故事生成模型
        generator = pipeline(
            "text-generation",
            model="pranavpsv/genre-story-generator-v2"
        )
        # 估算最大token数（英文约1.3 token/词）
        max_tokens = int(max_words * 1.3)
        # 生成故事
        result = generator(
            caption,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        story = result[0]['generated_text']
        
        # 如果故事仍然超过100词，进行截断
        words = story.split()
        if len(words) > max_words:
            story = ' '.join(words[:max_words]) + '...'
        
        return story
    except Exception as e:
        st.error(f"故事生成失败：{e}")
        return None

# ============================================================
# 文本转语音功能（保存为WAV文件）
# ============================================================
def text2audio(text, output_filename="story.wav"):
    """
    将文本转换为语音，并保存为WAV文件
    """
    try:
        # 加载TTS模型
        tts = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
        audio_data = tts(text)
        
        # 保存音频文件
        sf.write(output_filename, audio_data["audio"], samplerate=audio_data["sampling_rate"])
        return output_filename
    except Exception as e:
        st.error(f"语音合成失败：{e}")
        return None

# function part
def main():
    print("=== 图片讲故事应用 ===")
    filename = input("请输入图像文件名（图像需在当前目录）：").strip()
    if not os.path.exists(filename):
        print("文件不存在！")
        return
    print("\n1. 分析图像...")
    caption = img2text(filename)
    if not caption:
        return
    print(f"图像描述：{caption}")
    
    print("\n2. 生成故事...")
    story = text2story(caption, max_words=100)
    if not story:
        return
    print(f"\n故事：\n{story}")
    print(f"字数：{len(story.split())}")
    
    print("\n3. 合成语音...")
    audio_file = text2audio(story, output_filename="story.wav")
    if audio_file:
        print(f"音频已保存为 {audio_file}")
        # 可选播放（需要额外库）
        # import playsound
        # playsound.playsound(audio_file)

if __name__ == "__main__":
    main()
