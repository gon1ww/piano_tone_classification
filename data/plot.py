import librosa
import matplotlib.pyplot as plt
from modelscope.msdatasets import MsDataset
from tqdm import tqdm
import seaborn as sns

plt.rcParams["font.sans-serif"] = "Times New Roman"


def draw_pie_chart(
    labels: list,
    sizes: list,
    filename: str = "./data/piano.pdf",
    label_fontsize: int = 12,  # 修改标签字号
    autopct_fontsize: int = 12,  # 修改百分比字号
):
    """绘制饼图并保存为PDF文件"""
    _, _, autopcts = plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": label_fontsize},  # 设置标签字号
    )

    for autopct in autopcts:
        autopct.set_fontsize(autopct_fontsize)  # 设置百分比字号

    plt.axis("equal")  # 确保饼图为圆形
    plt.savefig(filename, bbox_inches="tight")  # 保存图像
    plt.close()


def plot():
    """绘制数据集中钢琴类型的分布图"""
    # 加载数据集
    ds = MsDataset.load('ccmusic-database/pianos', subset_name='default')
    
    # 获取标签名称
    if hasattr(ds["test"].features, "label"):
        classes = ds["test"].features["label"].names
    else:
        # 从 dataset_infos.json 读取类别信息
        import json
        with open("dataset_infos.json", "r") as f:
            info = json.load(f)
            classes = info["default"]["features"]["label"]["names"]
    
    # 统计每个类别的样本数量
    train_counts = {}
    test_counts = {}
    val_counts = {}
    
    # 统计训练集
    for item in ds["train"]:
        label = item["label"]
        train_counts[label] = train_counts.get(label, 0) + 1
    
    # 统计测试集
    for item in ds["test"]:
        label = item["label"]
        test_counts[label] = test_counts.get(label, 0) + 1
        
    # 统计验证集
    for item in ds["validation"]:
        label = item["label"]
        val_counts[label] = val_counts.get(label, 0) + 1
    
    # 绘图
    plt.figure(figsize=(15, 6))
    x = range(len(classes))
    
    plt.bar([i-0.2 for i in x], [train_counts.get(i, 0) for i in range(len(classes))], 
            width=0.2, label='Train', alpha=0.8)
    plt.bar([i for i in x], [test_counts.get(i, 0) for i in range(len(classes))], 
            width=0.2, label='Test', alpha=0.8)
    plt.bar([i+0.2 for i in x], [val_counts.get(i, 0) for i in range(len(classes))], 
            width=0.2, label='Validation', alpha=0.8)
    
    plt.xticks(x, classes, rotation=45)
    plt.xlabel('Piano Types')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Piano Types in Dataset')
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('./data/distribution.png')
    plt.close()
    
    return ds


if __name__ == "__main__":
    ds = plot()  # 执行绘图函数
