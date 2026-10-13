from __future__ import annotations

import argparse
import json
import random
import os, re
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

from utils.config import TaxonomyClasses

RE_DIGIT = re.compile(r"\d+")


class AnchorManager:
    """加载并解析形态锚点配置"""
    def __init__(self, anchors_yaml: Optional[str | Path] = None):
        self.orders: Dict[str, str] = {}
        self.families: Dict[str, str] = {}
        self.fallback: str = "insect"
        if anchors_yaml and Path(anchors_yaml).is_file():
            with open(anchors_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.orders = {k.lower().strip(): v for k, v in data.get("orders", {}).items()}
                self.families = {k.lower().strip(): v for k, v in data.get("families", {}).items()}
                self.fallback = data.get("fallback", "insect")

    def resolve(self, family: Optional[str], order: Optional[str]) -> str:
        f_key = family.strip().lower() if family else ""
        if f_key and f_key in self.families:
            return self.families[f_key]
        o_key = order.strip().lower() if order else ""
        if o_key and o_key in self.orders:
            return self.orders[o_key]
        return self.fallback


class BaseDatasetLoader:
    """负责解析标注 CSV (image_name, class_id) 与类别映射 class_map.json"""
    def __init__(self, dataset_csv: str | Path, class_map_json: str | Path):
        self.dataset_csv = Path(dataset_csv)
        self.class_map_json = Path(class_map_json)

        if not self.dataset_csv.is_file():
            raise FileNotFoundError(f"数据集标注 CSV 不存在: {self.dataset_csv}")
        if not self.class_map_json.is_file():
            raise FileNotFoundError(f"类别映射 JSON 不存在: {self.class_map_json}")

        self.id_to_alias: Dict[int, str] = self._load_class_map()
        self.records: List[Tuple[str, int, str]] = self._load_dataset()

    def _load_class_map(self) -> Dict[int, str]:
        with open(self.class_map_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        id_map: Dict[int, str] = {}
        for k, v in raw_map.items():
            try:
                id_map[int(k)] = str(v).strip()
            except ValueError:
                id_map[int(v)] = str(k).strip()
        id_map[-1] = "negative"
        return id_map

    def _load_dataset(self) -> List[Tuple[str, int, str]]:
        import pandas as pd
        df = pd.read_csv(self.dataset_csv)
        samples = []
        for _, row in df.iterrows():
            img_name = str(row["image_name"]).strip()
            class_id = int(row["class_id"])
            alias = self.id_to_alias.get(class_id, "negative" if class_id == -1 else "")
            if alias:
                samples.append((img_name, class_id, alias))
        return samples

    def get_positive_aliases(self) -> Set[str]:
        return {alias.lower() for _, class_id, alias in self.records if class_id != -1}


class KnowledgeBaseLoader:
    """加载解剖特征 JSON 知识库，并建立 科(Family) -> 物种拉丁名 的倒排索引"""

    def __init__(self, json_paths: List[str]):
        self.json_paths = [Path(p) for p in json_paths]
        self.knowledge: Dict[str, dict] = {}
        self.family_to_species: Dict[str, List[str]] = {}
        self._load_and_index()

    def _load_and_index(self):
        for path in self.json_paths:
            if not path.is_file():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    latin_key = k.lower().strip()
                    self.knowledge[latin_key] = v

        for latin, info in self.knowledge.items():
            family = str(info.get("family", "")).strip().lower()
            if not family:
                family = str(info.get("taxon_group", "")).strip().lower()
            if family:
                self.family_to_species.setdefault(family, []).append(latin)

    def get(self, latin_name: str) -> Optional[dict]:
        return self.knowledge.get(latin_name.lower().strip())

    def get_family(self, latin_name: str) -> Optional[str]:
        info = self.get(latin_name)
        if not info:
            return None
        family = str(info.get("family", "")).strip().lower()
        if not family:
            family = str(info.get("taxon_group", "")).strip().lower()
        return family or None


class CaptionSynthesizer:
    """均质化 Caption 发生器"""

    def __init__(
        self,
        kb_loader: KnowledgeBaseLoader,
        anchor_mgr: AnchorManager,
        num_dim_range: Tuple[int, int] = (2, 3),
        diagnostic_prob: float = 0.4,
        prompt_prefix: str = "A photo of a",
    ):
        self.kb_loader = kb_loader
        self.anchor_mgr = anchor_mgr
        self.num_dim_range = num_dim_range
        self.diagnostic_prob = diagnostic_prob
        self.prompt_prefix = prompt_prefix

    def generate(self, alias: str, latin_name: Optional[str], num_captions: int = 4) -> List[str]:
        target_name = latin_name or alias
        if not latin_name:
            return [f"{self.prompt_prefix} {target_name}." for _ in range(num_captions)]

        info = self.kb_loader.get(latin_name)
        if not info:
            return [f"{self.prompt_prefix} {target_name}." for _ in range(num_captions)]

        family = info.get("family", "") or info.get("taxon_group", "")
        order = info.get("order", "")
        anchor = self.anchor_mgr.resolve(family, order)

        results: List[str] = []
        # 所有 Caption 生成均采取等熵多维度自然句组合
        for _ in range(num_captions):
            base_phrase = BaseCaptionFormatter.format_base_identity(
                self.prompt_prefix, latin_name, anchor, family=family, order=order
            )
            dim_phrases = BaseCaptionFormatter.sample_dimensions(
                info.get("dimensions", {}), self.num_dim_range
            )
            
            parts = [base_phrase] + dim_phrases
            if random.random() < self.diagnostic_prob:
                diag = BaseCaptionFormatter.sample_diagnostic(info)
                if diag:
                    parts.append(diag)

            results.append(" ".join(parts))

        return results


class BaseCaptionFormatter:
    """提供通用的解剖维度拼接与兜底语句格式化工具"""
    @staticmethod
    def format_taxonomy(
        prompt_prefix: str,
        latin_name: str,
        family: Optional[str] = None,
        order: Optional[str] = None,
    ) -> str:
        """格式化分类学阶元，严格杜绝括号语法"""
        family_clean = family.strip() if family else ""
        order_clean = order.strip() if order else ""

        if family_clean and order_clean:
            return f"{prompt_prefix} {latin_name}, belonging to family {family_clean}, order {order_clean}."
        if family_clean:
            return f"{prompt_prefix} {latin_name}, belonging to family {family_clean}."
        if order_clean:
            return f"{prompt_prefix} {latin_name}, belonging to order {order_clean}."
        return f"{prompt_prefix} {latin_name}."

    @staticmethod
    def sample_dimensions(dims: dict, num_range: Tuple[int, int]) -> List[str]:
        """随机采样指定数量的解剖维度，并格式化为首字母大写、句尾带点的独立句子"""
        available_dims = [k for k, v in dims.items() if v]
        k_sample = min(len(available_dims), random.randint(num_range[0], num_range[1]))
        selected_dims = random.sample(available_dims, k_sample) if available_dims else []

        dim_phrases = []
        for dk in selected_dims:
            raw = dims[dk]
            phrase = random.choice(raw) if isinstance(raw, list) else str(raw)
            phrase = phrase.strip().rstrip(".,;")
            if phrase:
                dim_phrases.append(phrase[0].upper() + phrase[1:] + ".")
        return dim_phrases

    @staticmethod
    def format_base_identity(
        prompt_prefix: str,
        latin_name: str,
        anchor: str,
        family: Optional[str] = None,
        order: Optional[str] = None,
    ) -> str:
        """生成带自然形态锚点的基底引导短语"""
        fam = family.strip() if family else ""
        ord_ = order.strip() if order else ""

        if fam and ord_:
            return f"{prompt_prefix} {latin_name}, a {anchor} in the family {fam}, order {ord_}."
        if fam:
            return f"{prompt_prefix} {latin_name}, a {anchor} belonging to family {fam}."
        if ord_:
            return f"{prompt_prefix} {latin_name}, a {anchor} belonging to order {ord_}."
        return f"{prompt_prefix} {latin_name}, a {anchor}."

    @staticmethod
    def sample_diagnostic(info: dict) -> Optional[str]:
        diags = info.get("diagnostic_focus", [])
        if not diags:
            return None

        raw = random.choice(diags).strip().rstrip(".,;")
        if not raw:
            return None

        # 确保短语首字母为小写，便于拼接过渡副词
        phrase = raw[0].lower() + raw[1:] if len(raw) > 1 else raw.lower()

        natural_templates = [
            f"Notably, {phrase}.",
            f"In particular, {phrase}.",
            f"Characteristically, {phrase}.",
            f"As a defining feature, {phrase}.",
            f"{phrase[0].upper() + phrase[1:]}.",
        ]
        return random.choice(natural_templates)

    @staticmethod
    def make_fallback(target_name: str, prompt_prefix: str, num_captions: int) -> List[str]:
        templates = [
            f"{prompt_prefix} {target_name}.",
            f"A detailed photograph of {target_name}.",
            f"A close-up shot of {target_name}.",
            f"A clear photograph showing {target_name}.",
        ]
        return [random.choice(templates) for _ in range(num_captions)]


class NegativeAliasExtractor:
    """文件名解析 alias"""
    @staticmethod
    def extract_from_filename(img_name: str) -> str:
        stem = Path(img_name).stem
        for part in reversed(stem.split("_")):
            cleaned = RE_DIGIT.sub("", part).strip()
            if cleaned:
                return cleaned.lower()
        return "negative"


class LMDBStorageWriter:
    """将图像写入 LMDB，以 '{alias}+{img_name}' 为 Key"""

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
    """样本划分及 JSON 清单导出"""

    def __init__(self, ratios: List[float], seed: int = 42):
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-5:
            raise ValueError(f"ratios 长度必须为 3 且和为 1.0: {ratios}")
        self.ratios = ratios
        self.seed = seed

    def split(
        self, records: List[Tuple[str, List[str]]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Optional[Dict[str, List[str]]]]:
        rng = random.Random(self.seed)

        # 按 alias 分组: composite_key 为 "{alias}+{img_name}"
        class_buckets: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
        for item in records:
            composite_key = item[0]
            alias = composite_key.split("+", 1)[0]
            class_buckets[alias].append(item)

        train_records, val_records, test_records = [], [], []

        # 类别内独立分层抽样
        for alias, items in class_buckets.items():
            rng.shuffle(items)
            n = len(items)

            if n == 1:
                # 孤本强制放入训练集
                train_records.extend(items)
                continue

            n_val = max(1, int(round(n * self.ratios[1])))
            n_test = int(round(n * self.ratios[2])) if self.ratios[2] > 0 else 0
            n_train = n - n_val - n_test

            # 极端边界防御
            if n_train <= 0:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

            train_records.extend(items[:n_train])
            val_records.extend(items[n_train : n_train + n_val])
            if n_test > 0:
                test_records.extend(items[n_train + n_val :])

        rng.shuffle(train_records)
        rng.shuffle(val_records)
        rng.shuffle(test_records)

        train_data = dict(train_records)
        val_data = dict(val_records)
        test_data = dict(test_records) if test_records else None

        return train_data, val_data, test_data

    @staticmethod
    def export_json(data: dict, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _worker_process_chunk(chunk_records, synthesizer, taxonomy, num_captions):
    """并发生成 Captions"""
    results = []
    for img_name, class_id, base_alias in chunk_records:
        if class_id == -1 or base_alias.lower() == "negative":
            actual_alias = NegativeAliasExtractor.extract_from_filename(img_name)
            is_neg = True
        else:
            actual_alias = base_alias
            is_neg = False

        latin = taxonomy.get_latin(actual_alias)
        captions = synthesizer.generate(actual_alias, latin, num_captions=num_captions)
        results.append((img_name, actual_alias, captions, is_neg))
    return results


class DatasetPipeline:
    """顶层管线：协调正负样本分流、物理文件检索与存储产出"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.img_root = Path(args.img_root)
        self.out_dir = Path(args.out_dir)

        self.taxonomy = TaxonomyClasses.from_csv(args.taxonomy_csv)
        self.base_loader = BaseDatasetLoader(args.dataset_csv, args.class_map_json)
        self.kb_loader = KnowledgeBaseLoader(args.caption_jsons)

        self.synthesizer = CaptionSynthesizer(
            kb_loader=self.kb_loader,
            anchor_mgr=AnchorManager(args.anchors_yaml),
            num_dim_range=(args.min_dims, args.max_dims),
            diagnostic_prob=args.diagnostic_prob,
            prompt_prefix=args.prompt_prefix,
        )

        self.splitter = DatasetSplitter(ratios=args.ratios, seed=args.seed)
        self.manifest_records: List[Tuple[str, List[str]]] = []
        self.image_records: List[Tuple[str, Path]] = []
        self.img_cache: Dict[str, Path] = {}

    def _build_image_cache(self):
        """单次预先遍历全量图像，建立内存哈希索引，避免循环内部多次 stat 磁盘"""
        print("[Indexing] 正在预建图像磁盘文件索引...")
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in self.img_root.rglob("*"):
            if p.suffix.lower() in valid_exts:
                self.img_cache[p.name] = p
                rel_key = f"{p.parent.name}/{p.name}"
                self.img_cache[rel_key] = p
        print(f"[Indexing] 内存索引构建完成，已收录文件数: {len(self.img_cache)}")

    def _locate_image(self, candidate_dirs: List[str], img_name: str) -> Optional[Path]:
        for d in candidate_dirs:
            key = f"{d}/{img_name}"
            if key in self.img_cache:
                return self.img_cache[key]
        return self.img_cache.get(img_name)

    def process(self):
        self._build_image_cache()

        records = self.base_loader.records
        num_workers = min(os.cpu_count() or 4, 16)
        chunk_size = (len(records) + num_workers - 1) // num_workers
        chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]

        print(f"[Processing] 启动 {num_workers} 个进程并行生成 Captions...")
        processed_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _worker_process_chunk,
                    chunk,
                    self.synthesizer,
                    self.taxonomy,
                    self.args.num_captions,
                )
                for chunk in chunks
            ]
            for f in tqdm(futures, desc="[Parallel Workers]"):
                processed_results.extend(f.result())

        missing_imgs = 0
        pos_count = 0
        neg_count = 0

        for img_name, actual_alias, captions, is_neg in tqdm(processed_results, desc="[Resolving Images]"):
            candidate_dirs = [actual_alias, "negative"] if is_neg else [actual_alias]
            img_file = self._locate_image(candidate_dirs, img_name)

            if img_file is None:
                missing_imgs += 1
                continue

            if is_neg:
                neg_count += 1
            else:
                pos_count += 1

            composite_key = f"{actual_alias}+{img_name}"
            self.manifest_records.append((composite_key, captions))
            if self.args.make_lmdb:
                self.image_records.append((composite_key, img_file))

        print(f"[Summary] 完成构建: 正样本 {pos_count} 条, 负样本 {neg_count} 条")
        if missing_imgs > 0:
            print(f"[Warning] 磁盘未命中物理图像数: {missing_imgs}")

    def export(self):
        train_data, val_data, test_data = self.splitter.split(self.manifest_records)
        try:
            import orjson
            def _dump_json(data, path: Path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(orjson.dumps(data))
        except ImportError:
            def _dump_json(data, path: Path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

        _dump_json(train_data, self.out_dir / "train.json")
        _dump_json(val_data, self.out_dir / "valid.json")
        print(f"[Export] train.json: {len(train_data)} 项 -> {self.out_dir / 'train.json'}")
        print(f"[Export] valid.json: {len(val_data)} 项 -> {self.out_dir / 'valid.json'}")

        if test_data is not None:
            _dump_json(test_data, self.out_dir / "test.json")
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
    parser = argparse.ArgumentParser("细粒度害虫多模态数据集构建工具")
    parser.add_argument("--img_root", required=True, help="图像存放根目录")
    parser.add_argument("--dataset_csv", required=True, help="分类标注 CSV (含 image_name, class_id)")
    parser.add_argument("--class_map_json", required=True, help="映射文件 class_map.json (映射 class_id 与 alias)")
    parser.add_argument("--taxonomy_csv", required=True, help="物种名映射表 (含 index, pest_name, alias/pest_cname, pest_latin_name)")
    parser.add_argument("--caption_jsons", nargs="+", required=True, help="细粒度解剖特征 JSON 库路径列表")
    parser.add_argument("--anchors_yaml", default="configs/anchors.yaml", help="形态锚点映射文件路径")
    parser.add_argument("--out_dir", required=True, help="输出分割 JSON 目标文件夹")

    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="划分比例 [train, valid, test]")
    parser.add_argument("--num_captions", type=int, default=2, help="单张图像生成的描述数量")
    parser.add_argument("--diagnostic_prob", type=float, default=0.5, help="排他诊断特征触发概率")
    parser.add_argument("--min_dims", type=int, default=3, help="采样解剖维度的最小数量")
    parser.add_argument("--max_dims", type=int, default=4, help="采样解剖维度的最大数量")
    parser.add_argument("--prompt_prefix", type=str, default="A photo of a", help="Prompt 通用前缀")

    parser.add_argument("--make_lmdb", action="store_true", help="是否打包单文件 LMDB")
    parser.add_argument("--lmdb_out", default=None, help="LMDB 保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    pipeline = DatasetPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
