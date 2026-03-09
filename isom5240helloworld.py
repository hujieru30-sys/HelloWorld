# Program title: Storytelling App

# install required packages
!pip install streamlit transformers torch soundfile pillow

# import part
import streamlit as st
from transformers import pipeline
import soundfile as sf
import os
import tempfile

# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_text = pipe(text)[0]['generated_text']
    return story_text

# text2audio
def text2audio(story_text, output_filename="story.wav"):
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = pipe(story_text)
    return audio_data
    # save audio
    sf.write(output_filename, audio_data["audio"], samplerate=audio_data["sampling_rate"])
        return output_filename

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
