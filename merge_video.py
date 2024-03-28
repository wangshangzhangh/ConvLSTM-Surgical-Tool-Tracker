import os
import re
from subprocess import call

def merge_large_videos(folder_path):
    # 用于存储视频文件的列表
    video_files = []

    # 正则表达式，用于解析文件名
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{6})_VID(\d{3}).mp4')

    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):  # 确保是MP4文件
            file_path = os.path.join(folder_path, file)
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 大于10MB的文件
                match = pattern.match(file)
                if match:
                    # 将文件和VID编号添加到列表
                    video_files.append((file, int(match.group(3))))

    # 按VID编号排序
    video_files.sort(key=lambda x: x[1])

    # 创建一个文件列表字符串，用于FFmpeg命令
    file_list_path = os.path.join(folder_path, "file_list.txt")
    with open(file_list_path, 'w') as f:
        for file, _ in video_files:
            f.write(f"file '{os.path.join(folder_path, file)}'\n")

    # 设置输出文件名
    output_file = os.path.join(folder_path, "merged_video.mp4")

    # 使用FFmpeg合并视频
    call(["ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path, "-c", "copy", output_file])

    # 删除文件列表
    os.remove(file_list_path)

# 使用示例
merge_large_videos('/home/hp/data/xufeng/北医三院/修1/2023-8-2/2023-07-27_084206')
