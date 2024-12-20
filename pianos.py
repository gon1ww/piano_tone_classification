import os
import random
import datasets
from datasets.tasks import ImageClassification

'''
_NAMES：是一个字典，将钢琴品牌名称（如 PearlRiver、YoungChang 等）映射
到包含四个评分值（可能是不同维度的评分，具体需结合业务场景判断）的列表，用于后
续给数据集中的样本关联相应的评分信息。
'''
_NAMES = {
    "PearlRiver": [2.33, 2.53, 2.37, 2.41],
    "YoungChang": [2.53, 2.63, 2.97, 2.71],
    "Steinway-T": [3.6, 3.63, 3.67, 3.63],
    "Hsinghai": [3.4, 3.27, 3.2, 3.29],
    "Kawai": [3.17, 2.5, 2.93, 2.87],
    "Steinway": [4.23, 3.67, 4, 3.97],
    "Kawai-G": [3.37, 2.97, 3.07, 3.14],
    "Yamaha": [3.23, 3.03, 3.17, 3.14],
}

'''
_HOMEPAGE：通过结合当前文件名称构建数据集的主页链接，形式为
 https://www.modelscope.cn/datasets/ccmusic-database/ 加上当前文件名（去掉 .py 后缀）。
'''
_HOMEPAGE = f"https://www.modelscope.cn/datasets/ccmusic-database/{os.path.basename(__file__)[:-3]}"

'''
_DOMAIN：在 _HOMEPAGE 基础上进一步构建了一个域名相关的路径，用于后续拼接具体资源（如音频、梅尔频谱图等文件）的下载链接。
'''
_DOMAIN = f"{_HOMEPAGE}/resolve/master/data"

'''
_PITCHES：是一个大型字典，将特定的编码（如 009、010 等）映射到对应
的音高表示（如 A2、A2#/B2b 等），用于根据文件名等信息提取音高数据关
联到数据集样本中。
'''
_PITCHES = {
    "009": "A2",
    "010": "A2#/B2b",
    "011": "B2",
    "100": "C1",
    "101": "C1#/D1b",
    "102": "D1",
    "103": "D1#/E1b",
    "104": "E1",
    "105": "F1",
    "106": "F1#/G1b",
    "107": "G1",
    "108": "G1#/A1b",
    "109": "A1",
    "110": "A1#/B1b",
    "111": "B1",
    "200": "C",
    "201": "C#/Db",
    "202": "D",
    "203": "D#/Eb",
    "204": "E",
    "205": "F",
    "206": "F#/Gb",
    "207": "G",
    "208": "G#/Ab",
    "209": "A",
    "210": "A#/Bb",
    "211": "B",
    "300": "c",
    "301": "c#/db",
    "302": "d",
    "303": "d#/eb",
    "304": "e",
    "305": "f",
    "306": "f#/gb",
    "307": "g",
    "308": "g#/ab",
    "309": "a",
    "310": "a#/bb",
    "311": "b",
    "400": "c1",
    "401": "c1#/d1b",
    "402": "d1",
    "403": "d1#/e1b",
    "404": "e1",
    "405": "f1",
    "406": "f1#/g1b",
    "407": "g1",
    "408": "g1#/a1b",
    "409": "a1",
    "410": "a1#/b1b",
    "411": "b1",
    "500": "c2",
    "501": "c2#/d2b",
    "502": "d2",
    "503": "d2#/e2b",
    "504": "e2",
    "505": "f2",
    "506": "f2#/g2b",
    "507": "g2",
    "508": "g2#/a2b",
    "509": "a2",
    "510": "a2#/b2b",
    "511": "b2",
    "600": "c3",
    "601": "c3#/d3b",
    "602": "d3",
    "603": "d3#/e3b",
    "604": "e3",
    "605": "f3",
    "606": "f3#/g3b",
    "607": "g3",
    "608": "g3#/a3b",
    "609": "a3",
    "610": "a3#/b3b",
    "611": "b3",
    "700": "c4",
    "701": "c4#/d4b",
    "702": "d4",
    "703": "d4#/e4b",
    "704": "e4",
    "705": "f4",
    "706": "f4#/g4b",
    "707": "g4",
    "708": "g4#/a4b",
    "709": "a4",
    "710": "a4#/b4b",
    "711": "b4",
    "800": "c5",
}

'''
_URLS：定义了不同类型数据（音频、梅尔频谱图、评估相关数据）的下载链接模
板，通过 _DOMAIN 拼接具体文件名后缀形成完整的可下载链接。
'''
_URLS = {
    "audio": f"{_DOMAIN}/audio.zip",
    "mel": f"{_DOMAIN}/mel.zip",
    "eval": f"{_DOMAIN}/eval.zip",
}


class pianos(datasets.GeneratorBasedBuilder):
    def _info(self):
        names = list(_NAMES.keys())
        if self.config.name == "default":
            names = names[:-1]

        return datasets.DatasetInfo(
            features=(
                datasets.Features(
                    {
                        "audio": datasets.Audio(sampling_rate=44100),
                        "mel": datasets.Image(),
                        "label": datasets.features.ClassLabel(names=names),
                        "pitch": datasets.features.ClassLabel(
                            names=list(_PITCHES.values())
                        ),
                        "bass_score": datasets.Value("float32"),
                        "mid_score": datasets.Value("float32"),
                        "treble_score": datasets.Value("float32"),
                        "avg_score": datasets.Value("float32"),
                    }
                )
                if self.config.name != "eval"
                else datasets.Features(
                    {
                        "mel": datasets.Image(),
                        "label": datasets.features.ClassLabel(names=names),
                        "pitch": datasets.features.ClassLabel(
                            names=list(_PITCHES.values())
                        ),
                        "bass_score": datasets.Value("float32"),
                        "mid_score": datasets.Value("float32"),
                        "treble_score": datasets.Value("float32"),
                        "avg_score": datasets.Value("float32"),
                    }
                )
            ),
            homepage=_HOMEPAGE,
            license="CC-BY-NC-ND",
            version="1.2.0",
            supervised_keys=("mel", "label"),
            task_templates=ImageClassification(
                image_column="mel",
                label_column="label",
            ),
        )

    def _split_generators(self, dl_manager):
        dataset = []
        if self.config.name != "eval":
            subset = {}
            audio_files = dl_manager.download_and_extract(_URLS["audio"])
            for path in dl_manager.iter_files([audio_files]):
                fname = os.path.basename(path)
                if fname.endswith(".wav"):
                    lebal = os.path.basename(os.path.dirname(path))
                    if self.config.name == "default" and lebal == "Yamaha":
                        continue

                    subset[fname.split(".")[0]] = {
                        "audio": path,
                        "label": lebal,
                        "pitch": _PITCHES[fname[1:4]],
                        "bass_score": _NAMES[lebal][0],
                        "mid_score": _NAMES[lebal][1],
                        "treble_score": _NAMES[lebal][2],
                        "avg_score": _NAMES[lebal][3],
                    }

            mel_files = dl_manager.download_and_extract(_URLS["mel"])
            for path in dl_manager.iter_files([mel_files]):
                fname = os.path.basename(path)
                pname = fname.split(".")[0]
                if fname.endswith(".jpg") and pname in subset:
                    subset[pname]["mel"] = path

            dataset = list(subset.values())

        else:
            data_files = dl_manager.download_and_extract(_URLS["eval"])
            for path in dl_manager.iter_files([data_files]):
                fname: str = os.path.basename(path)
                if fname.endswith(".jpg"):
                    lebal = os.path.basename(os.path.dirname(path))
                    dataset.append(
                        {
                            "mel": path,
                            "label": lebal,
                            "pitch": _PITCHES[fname.split("_")[0]],
                            "bass_score": _NAMES[lebal][0],
                            "mid_score": _NAMES[lebal][1],
                            "treble_score": _NAMES[lebal][2],
                            "avg_score": _NAMES[lebal][3],
                        }
                    )

        names = list(_NAMES.keys())
        if self.config.name == "default":
            names = names[:-1]

        categories = {}
        for name in names:
            categories[name] = []

        for data in dataset:
            categories[data["label"]].append(data)

        testset, validset, trainset = [], [], []
        for cls in categories:
            random.shuffle(categories[cls])
            count = len(categories[cls])
            p80 = int(count * 0.8)
            p90 = int(count * 0.9)
            trainset += categories[cls][:p80]
            validset += categories[cls][p80:p90]
            testset += categories[cls][p90:]

        random.shuffle(trainset)
        random.shuffle(validset)
        random.shuffle(testset)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": trainset}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"files": validset}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": testset}
            ),
        ]

    def _generate_examples(self, files):
        for i, path in enumerate(files):
            yield i, path
