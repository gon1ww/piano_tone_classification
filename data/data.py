import os
import time
import shutil
import librosa
import zipfile
import requests
import matplotlib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100  # 定义音频采样率


def failist(text_to_write, file_name="./data/failist.txt"):
    """记录错误信息到文件"""
    print(text_to_write)
    with open(file_name, "a", encoding="utf-8") as file:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        file.write(f"[{ts}]{text_to_write}\n")


def compress(folder_path, zip_file_path):
    """将指定文件夹压缩为ZIP文件"""
    if not os.path.exists(folder_path):
        failist(f"Error: Folder '{folder_path}' does not exist.")
        return
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=os.path.join(os.path.basename(folder_path), relative_path))

    print(f"Compression complete. ZIP file created at: {zip_file_path}")


def extract_zip(zip_file_path, extract_folder):
    """解压缩ZIP文件到指定文件夹"""
    if not os.path.exists(zip_file_path):
        failist(f"Error: ZIP file '{zip_file_path}' does not exist.")
        return
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    print(f"Extraction complete. Files extracted to: {extract_folder}")


def download_zip(url, save_path):
    """从指定URL下载ZIP文件并保存到本地"""
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))
        progress_bar.close()
        print(f"Download complete. File saved to: {save_path}")
    else:
        failist(f"Error: Unable to download file. HTTP status code: {response.status_code}")


def find_wav_files(directory):
    """查找指定目录下的所有WAV文件"""
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


def wav2mel(y, sr: int, audio_path: str):
    """将WAV音频转换为梅尔频谱图并保存为图像"""
    mel_path = os.path.dirname(audio_path).replace("/audio", "/mel")
    os.makedirs(mel_path, exist_ok=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    plt.axis("off")
    plt.imsave(f"{mel_path}/{os.path.basename(audio_path).replace('.wav', '.jpg')}", log_mel_spec)
    plt.close()


def wav2eval(y, sr: int, audio_path: str, slient_filter=False, width=0.18):
    """将WAV音频转换为评估图像并保存"""
    mel_path = os.path.dirname(audio_path).replace("/audio", "/eval")
    os.makedirs(mel_path, exist_ok=True)
    pitch = os.path.basename(audio_path)[1:-4]
    non_silent = y
    if slient_filter:
        non_silents = librosa.effects.split(y, top_db=40)
        non_silent = np.concatenate([y[start:end] for start, end in non_silents])

    mel_spec = librosa.feature.melspectrogram(y=non_silent, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    dur = librosa.get_duration(y=non_silent, sr=sr)
    total_frames = log_mel_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_mel_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(f"{mel_path}/{pitch}_{round(dur, 2)}_{i}.jpg", bbox_inches="tight", pad_inches=0.0)
        plt.close()


def split_by_cpu(items):
    """将数据分割为多个批次以便于多进程处理"""
    num_cpus = os.cpu_count() - 1
    if num_cpus is None or num_cpus < 1:
        num_cpus = 1

    index = 0
    if type(items) == dict:
        split_items = [{} for _ in range(num_cpus)]
        for key, value in items.items():
            split_items[index][key] = value
            index = (index + 1) % num_cpus
    else:
        split_items = [[] for _ in range(num_cpus)]
        for item in items:
            split_items[index].append(item)
            index = (index + 1) % num_cpus

    return split_items, num_cpus


def batch_convert(wav_files: list):
    """批量转换WAV文件为梅尔频谱图和评估图像"""
    for wav_file in tqdm(wav_files, desc="Converting wav to jpg..."):
        try:
            y, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
            wav2mel(y, sr, wav_file)
            wav2eval(y, sr, wav_file)
        except Exception as e:
            failist(f"Error converting {wav_file} : {e}")


def multi_batch_convert(zipath="./data/audio.zip", multi=True):
    """从ZIP文件中提取音频并进行批量转换"""
    if not os.path.exists(zipath) and not os.path.exists("./data/audio"):
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.utils.constant import DownloadMode
        
        snapshot_download(
            'ccmusic-database/pianos',
            cache_dir='./data',
            revision='master'
        )
    
    if not os.path.exists("./data/audio"):
        extract_zip(zipath, "./data")

    wav_files = find_wav_files("./data/audio")
    if multi:
        batches, num_cpu = split_by_cpu(wav_files)
        with Pool(processes=num_cpu) as pool:
            pool.map(batch_convert, batches)
    else:
        batch_convert(wav_files)


def clean_cache():
    """清理缓存文件夹"""
    print("Cleaning caches...")
    if os.path.exists("./data/mel"):
        shutil.rmtree("./data/mel")

    if os.path.exists("./data/eval"):
        shutil.rmtree("./data/eval")

    if os.path.exists("./data/audio"):
        shutil.rmtree("./data/audio")


if __name__ == "__main__":
    matplotlib.use("Agg")  # 设置后端以便于保存图像
    clean_cache()  # 清理缓存
    multi_batch_convert()  # 执行批量转换
    compress("./data/mel", "./data/mel.zip")  # 压缩梅尔频谱图
    compress("./data/eval", "./data/eval.zip")  # 压缩评估图像
    clean_cache()  # 再次清理缓存
