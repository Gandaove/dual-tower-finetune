from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


class TaxonomyMapper:
    """解析物种元数据表 (包含 pest_name, pest_cname, pest_latin_name)"""

    def __init__(self, taxonomy_csv: str | Path):
        self.csv_path = Path(taxonomy_csv)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"物种映射表不存在: {self.csv_path}")
        self._cname_to_latin: Dict[str, str] = {}
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path)
        required_cols = {"pest_name", "pest_cname", "pest_latin_name"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"物种映射表缺失必要列: {missing}")

        for _, row in df.iterrows():
            cname = str(row["pest_cname"]).strip()
            latin = str(row["pest_latin_name"]).strip()
            if cname and latin and cname.lower() != "nan" and latin.lower() != "nan":
                self._cname_to_latin[cname] = latin

    def get_latin_name(self, cname: str) -> Optional[str]:
        return self._cname_to_latin.get(cname.strip())


class BaseDatasetLoader:
    """负责解析原分类任务的 dataset.csv (image_name, class_id) 与 class_map.json"""

    def __init__(self, dataset_csv: str | Path, class_map_json: str | Path):
        self.dataset_csv = Path(dataset_csv)
        self.class_map_json = Path(class_map_json)

        if not self.dataset_csv.is_file():
            raise FileNotFoundError(f"原数据集标注 CSV 不存在: {self.dataset_csv}")
        if not self.class_map_json.is_file():
            raise FileNotFoundError(f"类别映射 JSON 不存在: {self.class_map_json}")

        self.id_to_cname: Dict[int, str] = self._load_class_map()
        self.records: List[Tuple[str, str]] = self._load_dataset()

    def _load_class_map(self) -> Dict[int, str]:
        with open(self.class_map_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)

        id_map = {}
        for k, v in raw_map.items():
            # 兼容 {"0": "cname"} 或 {"cname": 0}
            if str(k).isdigit():
                id_map[int(k)] = str(v).strip()
            elif str(v).isdigit():
                id_map[int(v)] = str(k).strip()
            else:
                raise ValueError(f"class_map.json 格式无法识别为 key-id 对应: {k}: {v}")
        return id_map

    def _load_dataset(self) -> List[Tuple[str, str]]:
        df = pd.read_csv(self.dataset_csv)
        required_cols = {"image_name", "class_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"原数据集 CSV 缺失必要列: {missing}")

        samples = []
        for _, row in df.iterrows():
            img_name = str(row["image_name"]).strip()
            class_id = int(row["class_id"])
            cname = self.id_to_cname.get(class_id)
            if not cname:
                continue
            samples.append((img_name, cname))
        return samples


class KnowledgeBaseLoader:
    """负责聚合 1 到多个 Caption 属性描述 JSON 库"""

    def __init__(self, json_paths: List[str]):
        self.json_paths = [Path(p) for p in json_paths]
        self.knowledge: Dict[str, dict] = {}
        self._merge()
        self.all_species: List[str] = sorted(
            [k.strip().lower() for k in self.knowledge.keys() if k.strip()],
            key=lambda x: len(x),
            reverse=True,
        )

    def _merge(self):
        for path in self.json_paths:
            if not path.is_file():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    self.knowledge[k.lower().strip()] = v

    def get(self, latin_name: str) -> Optional[dict]:
        return self.knowledge.get(latin_name.lower().strip())


class CaptionGenerator:
    """基于拉丁名、科目、细粒度维度特征及混淆对比项合成自然语言描述。"""

    def __init__(
        self,
        kb_loader: KnowledgeBaseLoader,
        num_dim_range: Tuple[int, int] = (3, 4),
        contrast_prob: float = 0.5,
        simple_prob: float = 0.3,   # 极简模板分支概率
    ):
        self.kb_loader = kb_loader
        self.num_dim_range = num_dim_range
        self.contrast_prob = contrast_prob
        self.simple_prob = simple_prob

    def _sanitize_contrast(self, phrase: str, target_species: str) -> str:
        """基于全局词表安全替换本物种与竞争物种实体，保留原始句式结构。"""
        # 1. 优先清洗本物种 (避免 'of is' 语病: 'of <self>' -> '')
        target_pat = rf"\bof\s+{re.escape(target_species)}\b"
        phrase = re.sub(target_pat, "", phrase, flags=re.IGNORECASE)
        # 单独出现的本物种替换为代称
        phrase = re.sub(rf"\b{re.escape(target_species)}\b", "this species", phrase, flags=re.IGNORECASE)

        # 2. 匹配并替换所有竞争物种
        competing_species = [s for s in self.kb_loader.all_species if s != target_species]
        if competing_species:
            comp_pattern = re.compile(
                r"\b(" + "|".join(re.escape(s) for s in competing_species) + r")\b",
                flags=re.IGNORECASE,
            )
            phrase = comp_pattern.sub("related species", phrase)

        # 3. 规整多余空格并规范首字母与标点
        phrase = re.sub(r"\s+", " ", phrase).strip()
        phrase = phrase.rstrip(".") + "."
        return phrase[0].upper() + phrase[1:]

    def _sample_single(self, latin_name: str) -> str:
        info = self.kb_loader.get(latin_name)
        if not info or random.random() < self.simple_prob:
            return f"A photo of {latin_name}."

        taxon = info.get("taxon_group", "").strip()
        prefix = f"A photo of {latin_name}, belonging to {taxon}." if taxon else f"A photo of {latin_name}."

        dims = info.get("dimensions", {})
        available_dims = [k for k, v in dims.items() if v]

        # 严格限制采样维度数量
        k_sample = min(len(available_dims), random.randint(self.num_dim_range[0], self.num_dim_range[1]))
        selected_dims = random.sample(available_dims, k_sample) if available_dims else []

        dim_phrases = []
        for dk in selected_dims:
            raw_phrase = random.choice(dims[dk]).strip()
            if raw_phrase:
                raw_phrase = raw_phrase.rstrip(".") + "."
                dim_phrases.append(raw_phrase[0].upper() + raw_phrase[1:])

        contrast_phrase = ""
        contrasts = info.get("confusion_contrasts", [])
        if contrasts and random.random() < self.contrast_prob:
            raw_c = random.choice(contrasts).strip()
            if raw_c:
                contrast_phrase = self._sanitize_contrast(raw_c, latin_name.lower().strip())

        parts = [prefix] + dim_phrases
        if contrast_phrase:
            parts.append(contrast_phrase)

        return " ".join(parts)

    def generate(self, latin_name: str, num_captions: int = 2) -> List[str]:
        return [self._sample_single(latin_name) for _ in range(num_captions)]


class LMDBStorageWriter:
    """负责将图像二进制数据写入 LMDB，以 '{中文全拼名}+{图像文件名}' 为 Key。"""

    def __init__(self, out_path: str | Path, map_size: int = 1024 * 1024 * 1024 * 50):
        self.out_path = Path(out_path)
        self.map_size = map_size

    def write(self, records: List[Tuple[str, Path]], commit_interval: int = 2000):
        import lmdb
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            str(self.out_path),
            subdir=False,
            map_size=self.map_size,
            readonly=False,
            meminit=False,
            map_async=True,
        )
        txn = env.begin(write=True)

        for idx, (db_key, img_path) in enumerate(tqdm(records, desc="[LMDB] Packing")):
            raw_bytes = img_path.read_bytes()
            txn.put(db_key.encode("utf-8"), raw_bytes)
            if (idx + 1) % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        env.sync()
        env.close()


class DatasetSplitter:
    """负责样本划分及生成对应的 JSON 清单。"""

    def __init__(self, ratios: List[float], seed: int = 42):
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-5:
            raise ValueError(f"ratios 长度必须为 3 且和为 1.0: {ratios}")
        self.ratios = ratios
        self.seed = seed

    def split(self, records: List[Tuple[str, List[str]]]) -> Tuple[dict, dict, Optional[dict]]:
        rng = random.Random(self.seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_tr = int(n_total * self.ratios[0])
        n_va = int(n_total * self.ratios[1])

        train_data = dict(shuffled[:n_tr])
        val_data = dict(shuffled[n_tr : n_tr + n_va])
        test_data = dict(shuffled[n_tr + n_va :]) if self.ratios[2] > 0 else None

        return train_data, val_data, test_data

    @staticmethod
    def export_json(data: dict, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class DatasetPipeline:
    """顶层协调管线，封装从图像扫描、文本合成到导出存储的完整业务流。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.img_root = Path(args.img_root)
        self.out_dir = Path(args.out_dir)

        # 实例化子模块
        self.tax_mapper = TaxonomyMapper(args.taxonomy_csv)
        self.base_loader = BaseDatasetLoader(args.dataset_csv, args.class_map_json)
        self.kb_loader = KnowledgeBaseLoader(args.caption_jsons)
        self.generator = CaptionGenerator(
            kb_loader=self.kb_loader,
            num_dim_range=(args.min_dims, args.max_dims),
            contrast_prob=args.contrast_prob,
            simple_prob=args.simple_prob,
        )
        self.splitter = DatasetSplitter(ratios=args.ratios, seed=args.seed)

        self.manifest_records: List[Tuple[str, List[str]]] = []
        self.image_records: List[Tuple[str, Path]] = []

    def _locate_image(self, cname: str, img_name: str) -> Optional[Path]:
        # 探测优先级 1: img_root / cname / img_name
        p1 = self.img_root / cname / img_name
        if p1.is_file():
            return p1
        # 探测优先级 2: img_root / img_name
        p2 = self.img_root / img_name
        if p2.is_file():
            return p2
        return None

    def process(self):
        missing_imgs = 0
        missing_tax = 0

        for img_name, cname in tqdm(self.base_loader.records, desc="[Matching] Records"):
            latin_name = self.tax_mapper.get_latin_name(cname)
            if not latin_name:
                print(f"not matched latin of {cname}")
                missing_tax += 1
                continue

            img_file = self._locate_image(cname, img_name)
            if img_file is None:
                missing_imgs += 1
                continue

            composite_key = f"{cname}+{img_name}"
            captions = self.generator.generate(latin_name, num_captions=self.args.num_captions)

            self.manifest_records.append((composite_key, captions))
            if self.args.make_lmdb:
                self.image_records.append((composite_key, img_file))

        print(f"[Summary] 成功构建样本数: {len(self.manifest_records)}")
        if missing_imgs > 0:
            print(f"[Warning] 未在磁盘找到物理图像数: {missing_imgs}")
        if missing_tax > 0:
            print(f"[Warning] 物种表未匹配到拉丁学名数: {missing_tax}")

    def export(self):
        train_data, val_data, test_data = self.splitter.split(self.manifest_records)
        DatasetSplitter.export_json(train_data, self.out_dir / "train.json")
        DatasetSplitter.export_json(val_data, self.out_dir / "valid.json")
        print(f"[Export] train.json: {len(train_data)} 项 -> {self.out_dir / 'train.json'}")
        print(f"[Export] valid.json: {len(val_data)} 项 -> {self.out_dir / 'valid.json'}")

        if test_data is not None:
            DatasetSplitter.export_json(test_data, self.out_dir / "test.json")
            print(f"[Export] test.json: {len(test_data)} 项 -> {self.out_dir / 'test.json'}")

        if self.args.make_lmdb:
            lmdb_path = self.args.lmdb_out or str(self.out_dir / "images.lmdb")
            writer = LMDBStorageWriter(lmdb_path)
            writer.write(self.image_records)
            print(f"[Export] LMDB 构建完成 -> {lmdb_path}")

    def run(self):
        self.process()
        self.export()


def main():
    parser = argparse.ArgumentParser("嫁接原分类标注的多模态数据集构建工具")
    # 路径参数明确区分
    parser.add_argument("--img_root", required=True, help="图像存放根目录")
    parser.add_argument("--dataset_csv", required=True, help="原分类数据集标注 CSV (必须含 image_name, class_id)")
    parser.add_argument("--class_map_json", required=True, help="原分类映射 class_map.json (映射 class_id 与 全拼名)")
    parser.add_argument("--taxonomy_csv", required=True, help="物种名映射表 (必须含 pest_name, pest_cname, pest_latin_name)")
    parser.add_argument("--caption_jsons", nargs="+", required=True, help="细粒度属性 Caption 知识库 JSON 列表")
    parser.add_argument("--out_dir", required=True, help="输出分割 JSON 的目标文件夹")

    # 控制参数
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="划分比例 [train, valid, test]")
    parser.add_argument("--num_captions", type=int, default=2, help="每个图像生成的 Caption 数量")
    parser.add_argument("--contrast_prob", type=float, default=0.5, help="引入易混淆对比描述的概率")
    parser.add_argument("--simple_prob", type=float, default=0.3, help="输出极简模板的概率")
    parser.add_argument("--min_dims", type=int, default=3, help="采样属性维度的最小数量")
    parser.add_argument("--max_dims", type=int, default=4, help="采样属性维度的最大数量")
    parser.add_argument("--make_lmdb", action="store_true", help="是否同时写入单文件 LMDB")
    parser.add_argument("--lmdb_out", default=None, help="LMDB 单文件保存路径 (默认: out_dir/images.lmdb)")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    args = parser.parse_args()

    pipeline = DatasetPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
