import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import requests
import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
from diffusers import StableDiffusionPipeline
import torch
from moviepy import *
import glob

# --- AIモデルのロード (MusicGen) ---

print("🎵 MusicGenモデルのロードを開始します... (時間がかかります)")
try:
    # デバイスの自動選択
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    model_id = "facebook/musicgen-small" 
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device)
    print("🎵 MusicGenモデルのロード完了")

except Exception as e:
    print(f"エラー: モデルのロードに失敗しました。{e}")
    model = None
    processor = None

# --- AIモデルのロード (Stable Diffusion) ---
print("🎨 Stable Diffusionモデルのロードを開始します... (時間がかかります)")
try:
    # デバイスの自動選択 (MusicGenと共通)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # モデルのロード
    model_id = "runwayml/stable-diffusion-v1-5" 
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32 
        # ↑ GPUなら高速なfloat16、CPUならfloat32
    ).to(device)
    
    print("🎨 Stable Diffusionモデルのロード完了")

except Exception as e:
    print(f"エラー: Stable Diffusionモデルのロードに失敗しました。{e}")
    pipe = None

VIDEO_DURATION_SECONDS = 8 # 動画の長さ（秒）
NUM_IMAGES = 5 # 生成する画像の枚数
FPS = 24 # 動画のフレームレート

# APIキー
OPENWEATHER_API_KEY = "1b29095b72c45b44d310f5e55afd6c49"

# FastAPIアプリの初期化
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. 入力データの型定義
class GenerationRequest(BaseModel):
    city: str
    text_prompt: str

# 2. OpenWeather APIから天気を取得する関数
def get_weather(city: str):
    """指定された都市の現在の天気を取得する"""
    if not OPENWEATHER_API_KEY:
        print("エラー: OPENWEATHER_API_KEYが設定されていません。")
        # デバッグ用にダミーデータを返す
        return {"weather": [{"main": "Clear", "description": "clear sky"}], "main": {"temp": 25}}

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric", # 温度を摂氏（℃）で取得
        "lang": "ja"      # 日本語で取得
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # HTTPエラーがあれば例外を発生
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"APIリクエストエラー: {e}")
        return None

# 3. AIモデルの呼び出し
def run_musicgen(prompt: str, duration_seconds: int) -> str:
    """MusicGenで音楽を生成する"""
    print(f"🎵 MusicGen実行: {prompt}")

    try:
        # プロンプトを準備
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # --- 動画の長さに合わせてトークン数を計算 ---
        # (この値は目安です。musicgen-small の場合)
        tokens_per_second = 50 
        max_tokens = int(duration_seconds * tokens_per_second)  

        # 音楽を生成 
        # もっと長くする場合は max_new_tokens を増やす (例: 1500 で 30秒)
        audio_values = model.generate(**inputs, max_new_tokens=max_tokens) 
        
        # ファイルに保存
        output_path = "static/music/generated_music.wav"
        sampling_rate = model.config.audio_encoder.sampling_rate
        
        # バッチ処理（今回は1つ）の最初の音声を取得
        audio_data = audio_values[0, 0].cpu().numpy()
        
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data)
        
        print(f"🎵 音楽を {output_path} に保存しました。")
        return output_path

    except Exception as e:
        print(f"MusicGen実行エラー: {e}")
        return "static/music/error_music.wav" # エラー用ファイルを返す
    
def run_stable_diffusion(prompt: str, num_images: int) -> list[str]:
    """Stable Diffusionで画像を生成する"""
    print(f"🎨 Stable Diffusion実行: {prompt}")
    
    if pipe is None:
        print("エラー: Stable Diffusionモデルがロードされていません。")
        return [f"static/images/dummy_image_{i}.png" for i in range(num_images)]

    try:
        generated_paths = []
        # プロンプト (ネガティブプロンプトも指定すると品質が上がることが多い)
        neg_prompt = "low quality, bad hands, blurry"
        
        # 高速化のための設定 (GPUの場合)
        if device == "cuda":
            pipe.enable_model_cpu_offload() # メモリが少ない場合に有効
            # pipe.enable_xformers_memory_efficient_attention() # xformers があれば

        for i in range(num_images):
            # 画像を生成
            image = pipe(
                prompt=prompt, 
                negative_prompt=neg_prompt,
                num_inference_steps=20 # ステップ数 (少ないと速いが荒い)
            ).images[0] # 最初の画像を取得
            
            # ファイルに保存
            output_path = f"static/images/generated_image_{i}.png"
            image.save(output_path)
            generated_paths.append(output_path)
            print(f"🎨 画像 {i+1}/{num_images} を {output_path} に保存しました。")
        
        return generated_paths

    except Exception as e:
        print(f"Stable Diffusion実行エラー: {e}")
        return [f"static/images/error_image_{i}.png" for i in range(num_images)]    
    

def create_video(music_path: str, image_paths: list[str], duration: int) -> str:
    """MoviePyを使って音楽と画像を動画に合成する"""
    print("🎬 動画を生成中です...")
    output_video_path = "static/video/final_video.mp4"

    try:
        # 1. 音楽ファイルを読み込む
        audio_clip = AudioFileClip(music_path)
        
        # 音楽が指定した動画長より短い場合、動画の長さを音楽に合わせる
        # (または、指定した長さでカットする)
        final_duration = min(audio_clip.duration, duration)
        audio_clip = audio_clip.subclipped(0, final_duration)

        # 2. 画像から動画クリップを作成
        image_clips = []
        # 1枚あたりの表示時間
        duration_per_image = final_duration / len(image_paths)

        for path in image_paths:
            # 0.5秒でフェードイン/アウトする設定
            img_clip = ImageClip(path).with_duration(duration_per_image).with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.5)])
            image_clips.append(img_clip)

        # 3. 画像クリップを順番に連結
        video_clip = concatenate_videoclips(image_clips, method="compose")

        # 4. 動画に音楽をセット
        final_video = video_clip.with_audio(audio_clip)

        # 5. MP4ファイルとして書き出す
        final_video.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            fps=FPS 
        )
        
        print(f"🎉 動画を {output_video_path} に保存しました！")
        return output_video_path

    except Exception as e:
        print(f"動画生成エラー: {e}")
        return "static/video/error_video.mp4" # エラー用パス (ダミー)
    
# 4. メインのAPIエンドポイント
@app.post("/generate")
async def generate_media(request: GenerationRequest):
    """天気とテキストから音楽と画像を生成する"""
    
    # (a) 天気を取得
    weather_data = get_weather(request.city)
    if not weather_data:
        return JSONResponse(status_code=500, content={"message": "天気情報の取得に失敗しました。"})
        
    weather_main = weather_data.get("weather", [{}])[0].get("main", "Unknown") # 'Rain', 'Clear', 'Clouds' など
    weather_desc = weather_data.get("weather", [{}])[0].get("description", "unknown weather")
    temperature = weather_data.get("main", {}).get("temp", "N/A")

    print(f"取得した天気: {weather_main} ({weather_desc}), 気温: {temperature}℃")

    # (b) AI用のプロンプトを生成
    music_prompt = f"{weather_main}, {weather_desc}, {request.text_prompt}, cinematic music"
    image_prompt = f"A beautiful cinematic scene of {request.city} during {weather_main}, {request.text_prompt}, photorealistic, 4k"

    # (c) AIモデルを実行
    music_file_path = run_musicgen(music_prompt, duration_seconds=VIDEO_DURATION_SECONDS)
    image_file_paths = run_stable_diffusion(image_prompt, num_images=NUM_IMAGES)
    video_file_path = create_video(music_file_path, image_file_paths, VIDEO_DURATION_SECONDS)

    # (d) 結果を返す
    return {
        "message": "生成が完了しました！",
        "weather_info": f"{request.city}: {weather_desc}, {temperature}℃",
        "music_url": music_file_path,
        "image_urls": image_file_paths,
        "video_url": video_file_path,
        "prompts": {
            "music": music_prompt,
            "image": image_prompt
        }
    }

# 5. UI（HTML）を返すエンドポイント
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """フロントエンドのHTMLを返す"""
    # 実際にはindex.htmlファイルを読み込むのが良いです
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# サーバーの起動 (デバッグ用)
if __name__ == "__main__":
    # staticディレクトリがない場合は作成
    os.makedirs("static/music", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("static/video", exist_ok=True)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
