from __future__ import annotations

import argparse
import json
import random
import os, re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

from utils.config import TaxonomyClasses

RE_DIGIT = re.compile(r"\d+")


class CompetitorSelector:
    """与混淆强度的相似类选择器"""
    def __init__(
        self,
        taxonomy: TaxonomyClasses,
        kb_loader: KnowledgeBaseLoader,
        confusion_json_path: Optional[str | Path] = None,
        hard_prob: float = 0.8,
    ):
        self.taxonomy = taxonomy
        self.kb_loader = kb_loader
        self.hard_prob = hard_prob

        # 当前存在的合法别名全集 (自动兼容类别增删)
        self.valid_aliases: Set[str] = set(self.taxonomy.alias_to_latin.keys())
        self.precomputed_pos: Dict[str, Tuple[List[str], List[float]]] = {}
        self.precomputed_neg: Dict[str, Tuple[List[str], List[float]]] = {}

        if confusion_json_path:
            p = Path(confusion_json_path)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                self._precompile_distributions(raw_data)

    def _precompile_distributions(self, raw_data: dict):
        """一次性预编译所有类别的候选拉丁名及采样权重"""
        compiled_count = 0
        for src_alias, targets in tqdm(raw_data.items(), desc="[Precompile] 混淆转移分布预编译"):
            src_clean = src_alias.strip().lower()
            if src_clean not in self.valid_aliases:
                continue

            pos_latins, pos_weights = [], []
            neg_latins, neg_weights = [], []

            for tgt_alias, score in targets.items():
                tgt_clean = tgt_alias.strip().lower()
                if tgt_clean not in self.valid_aliases or tgt_clean == src_clean:
                    continue

                latin = self.taxonomy.get_latin(tgt_clean)
                if not latin:
                    continue

                val = float(score)
                if self.taxonomy.is_pos(tgt_clean):
                    pos_latins.append(latin)
                    pos_weights.append(val)
                else:
                    neg_latins.append(latin)
                    neg_weights.append(val)

            if pos_latins:
                self.precomputed_pos[src_clean] = (pos_latins, pos_weights)
            if neg_latins:
                self.precomputed_neg[src_clean] = (neg_latins, neg_weights)

            compiled_count += 1
        print(f"[Info] 混淆转移分布预编译完成，有效映射物种数: {compiled_count}")

    def _fallback_by_family(self, current_latin: str, only_positive: bool = False) -> Optional[str]:
        """原版同科随机回退逻辑"""
        family = self.kb_loader.get_family(current_latin)
        if not family:
            return None

        all_in_family = set(self.kb_loader.family_to_species.get(family, []))
        all_in_family.discard(current_latin.lower().strip())

        # 获取当前 taxonomy 中所有正类的拉丁名
        pos_latins = set(self.taxonomy.pos_latins)
        pos_in_family = [s for s in all_in_family if s in pos_latins]

        if pos_in_family:
            return random.choice(pos_in_family)

        if not only_positive:
            other_in_family = list(all_in_family)
            if other_in_family:
                return random.choice(other_in_family)

        return None

    def select_for_positive(self, current_alias: str, current_latin: str) -> Optional[str]:
        alias_clean = current_alias.strip().lower()
        if random.random() < self.hard_prob:
            # 优先级 1: 混淆正类 (C 实现单次快速抽样)
            if alias_clean in self.precomputed_pos:
                cand, weights = self.precomputed_pos[alias_clean]
                return random.choices(cand, weights=weights, k=1)[0]
            # 优先级 2: 混淆负类
            if alias_clean in self.precomputed_neg:
                cand, weights = self.precomputed_neg[alias_clean]
                return random.choices(cand, weights=weights, k=1)[0]

        # 兜底回退
        return self._fallback_by_family(current_latin, only_positive=False)

    def select_for_negative(self, current_alias: str, current_latin: str) -> Optional[str]:
        alias_clean = current_alias.strip().lower()
        if random.random() < self.hard_prob:
            # 负类严格只与正类对比
            if alias_clean in self.precomputed_pos:
                cand, weights = self.precomputed_pos[alias_clean]
                return random.choices(cand, weights=weights, k=1)[0]

        return self._fallback_by_family(current_latin, only_positive=True)


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

    @staticmethod
    def _parse_int(val: str) -> Optional[int]:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _load_class_map(self) -> Dict[int, str]:
        with open(self.class_map_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)

        id_map: Dict[int, str] = {}
        for k, v in raw_map.items():
            k_int = self._parse_int(str(k))
            v_int = self._parse_int(str(v))
            if k_int is not None:
                id_map[k_int] = str(v).strip()
            elif v_int is not None:
                id_map[v_int] = str(k).strip()
        # 负样本固定标识
        id_map[-1] = "negative"
        return id_map

    def _load_dataset(self) -> List[Tuple[str, int, str]]:
        df = pd.read_csv(self.dataset_csv)
        required_cols = {"image_name", "class_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"数据集 CSV 缺失必要列: {missing}")

        samples = []
        for _, row in df.iterrows():
            img_name = str(row["image_name"]).strip()
            class_id = int(row["class_id"])
            alias = self.id_to_alias.get(class_id, "negative" if class_id == -1 else "")
            if not alias:
                continue
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


class BaseCaptionFormatter:
    """提供通用的解剖维度拼接与兜底语句格式化工具"""

    @staticmethod
    def sample_dimensions(dims: dict, num_range: Tuple[int, int]) -> List[str]:
        available_dims = [k for k, v in dims.items() if v]
        k_sample = min(len(available_dims), random.randint(num_range[0], num_range[1]))
        selected_dims = random.sample(available_dims, k_sample) if available_dims else []

        dim_phrases = []
        for dk in selected_dims:
            raw = dims[dk]
            phrase = random.choice(raw) if isinstance(raw, list) else str(raw)
            phrase = phrase.strip().rstrip(".")
            if phrase:
                dim_phrases.append(phrase[0].upper() + phrase[1:] + ".")
        return dim_phrases

    @staticmethod
    def make_fallback(target_name: str, prompt_prefix: str, num_captions: int) -> List[str]:
        templates = [
            f"{prompt_prefix} {target_name}.",
            f"A detailed photograph of {target_name}.",
            f"A close-up shot of {target_name}.",
            f"A clear photograph showing {target_name}.",
        ]
        return [random.choice(templates) for _ in range(num_captions)]

    @staticmethod
    def build_contrast_phrase(info: dict, competitor_latin: str) -> Optional[str]:
        diags = info.get("diagnostic_focus", [])
        if diags:
            diag = random.choice(diags).strip().rstrip(".")
            if diag:
                return f"Distinguished from {competitor_latin} by {diag}."

        contrasts = info.get("confusion_contrasts", [])
        if contrasts:
            raw_c = random.choice(contrasts).strip().rstrip(".")
            if raw_c:
                return f"Distinguished from {competitor_latin} by {raw_c}."
        return None


class PositiveCaptionGenerator:
    """正样本描述生成器：优先挑选同科正类物种进行区分，兜底可选同科其他物种"""

    def __init__(
        self,
        kb_loader: KnowledgeBaseLoader,
        competitor_selector: CompetitorSelector,
        num_dim_range: Tuple[int, int] = (3, 4),
        contrast_prob: float = 0.5,
        simple_prob: float = 0.3,
        prompt_prefix: str = "A photo of a",
    ):
        self.kb_loader = kb_loader
        self.selector = competitor_selector
        self.num_dim_range = num_dim_range
        self.contrast_prob = contrast_prob
        self.simple_prob = simple_prob
        self.prompt_prefix = prompt_prefix

    def generate(self, alias: str, latin_name: str, num_captions: int = 2) -> List[str]:
        info = self.kb_loader.get(latin_name)
        if not info:
            return BaseCaptionFormatter.make_fallback(latin_name or alias, self.prompt_prefix, num_captions)

        results = []
        for _ in range(num_captions):
            if random.random() < self.simple_prob:
                results.append(f"{self.prompt_prefix} {latin_name}.")
                continue

            order = info.get("order", "").strip()
            family = info.get("family", "").strip()
            if family and order:
                prefix = f"{self.prompt_prefix} {latin_name}, belonging to {family} ({order})."
            elif family:
                prefix = f"{self.prompt_prefix} {latin_name}, belonging to {family}."
            else:
                prefix = f"{self.prompt_prefix} {latin_name}."

            dim_phrases = BaseCaptionFormatter.sample_dimensions(info.get("dimensions", {}), self.num_dim_range)

            # 核心调用点：通过 selector 依据混淆权重挑选竞争对手
            contrast_phrase = None
            if random.random() < self.contrast_prob:
                competitor_latin = self.selector.select_for_positive(alias, latin_name)
                if competitor_latin:
                    contrast_phrase = BaseCaptionFormatter.build_contrast_phrase(info, competitor_latin)

            parts = [prefix] + dim_phrases
            if contrast_phrase:
                parts.append(contrast_phrase)
            results.append(" ".join(parts))

        return results


class NegativeSampleHandler:
    """负样本处理器：
    1. 从文件名提取真实 alias
    2. 映射拉丁名并抽取细粒度特征
    3. 区分性描述严格限制仅挑同科的正类物种，未命中则跳过对比分支
    4. 无法命中元数据时执行平滑兜底
    """

    def __init__(
        self,
        taxonomy: TaxonomyClasses,
        kb_loader: KnowledgeBaseLoader,
        competitor_selector: CompetitorSelector,
        num_dim_range: Tuple[int, int] = (2, 4),
        contrast_prob: float = 0.5,
        simple_prob: float = 0.2,
        prompt_prefix: str = "A photo of a",
    ):
        self.taxonomy = taxonomy
        self.kb_loader = kb_loader
        self.selector = competitor_selector
        self.num_dim_range = num_dim_range
        self.contrast_prob = contrast_prob
        self.simple_prob = simple_prob
        self.prompt_prefix = prompt_prefix

    @staticmethod
    def extract_alias_from_filename(img_name: str) -> str:
        """解析如 RF..._ALARM_INPUT_pingzhangzhoue1.jpg，提取 pingzhangzhoue"""
        stem = Path(img_name).stem
        parts = stem.split("_")
        # 从末尾倒序查找首个包含字母的有效分块
        for part in reversed(parts):
            cleaned = RE_DIGIT.sub("", part).strip()
            if cleaned:
                return cleaned.lower()
        return "negative"

    def process_sample(self, img_name: str, num_captions: int = 2) -> Tuple[str, List[str]]:
        alias = self.extract_alias_from_filename(img_name)
        latin_name = self.taxonomy.get_latin(alias)

        if not latin_name:
            fallback = BaseCaptionFormatter.make_fallback(alias, self.prompt_prefix, num_captions)
            return alias, fallback

        info = self.kb_loader.get(latin_name)
        if not info:
            fallback = BaseCaptionFormatter.make_fallback(latin_name, self.prompt_prefix, num_captions)
            return alias, fallback

        results = []
        for _ in range(num_captions):
            if random.random() < self.simple_prob:
                results.append(f"{self.prompt_prefix} {latin_name}.")
                continue

            family = info.get("family", "").strip()
            order = info.get("order", "").strip()
            prefix = f"{self.prompt_prefix} {latin_name}, belonging to {family} ({order})." if (family and order) else f"{self.prompt_prefix} {latin_name}."

            dim_phrases = BaseCaptionFormatter.sample_dimensions(info.get("dimensions", {}), self.num_dim_range)

            # 核心调用点：负类样本严格挑选正类混淆对手
            contrast_phrase = None
            if random.random() < self.contrast_prob:
                competitor_latin = self.selector.select_for_negative(alias, latin_name)
                if competitor_latin:
                    contrast_phrase = BaseCaptionFormatter.build_contrast_phrase(info, competitor_latin)

            parts = [prefix] + dim_phrases
            if contrast_phrase:
                parts.append(contrast_phrase)
            results.append(" ".join(parts))

        return alias, results


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


def _worker_process_chunk(chunk_records, pos_generator, neg_handler, taxonomy, num_captions):
    """并发生成 Captions"""
    results = []
    for img_name, class_id, base_alias in chunk_records:
        if class_id == -1 or base_alias.lower() == "negative":
            actual_alias, captions = neg_handler.process_sample(img_name, num_captions=num_captions)
            is_neg = True
        else:
            actual_alias = base_alias
            latin = taxonomy.get_latin(actual_alias)
            captions = pos_generator.generate(actual_alias, latin, num_captions=num_captions)
            is_neg = False

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

        self.competitor_selector = CompetitorSelector(
            taxonomy=self.taxonomy,
            kb_loader=self.kb_loader,
            confusion_json_path=getattr(args, "confusion_json", None),
            hard_prob=args.hard_prob,
        )

        self.pos_generator = PositiveCaptionGenerator(
            kb_loader=self.kb_loader,
            competitor_selector=self.competitor_selector,
            num_dim_range=(args.min_dims, args.max_dims),
            contrast_prob=args.contrast_prob,
            simple_prob=args.simple_prob,
            prompt_prefix=args.prompt_prefix,
        )

        self.neg_handler = NegativeSampleHandler(
            taxonomy=self.taxonomy,
            kb_loader=self.kb_loader,
            competitor_selector=self.competitor_selector,
            num_dim_range=(args.min_dims, args.max_dims),
            contrast_prob=args.contrast_prob,
            simple_prob=args.simple_prob,
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
                    self.pos_generator,
                    self.neg_handler,
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
    parser.add_argument("--out_dir", required=True, help="输出分割 JSON 目标文件夹")

    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="划分比例 [train, valid, test]")
    parser.add_argument("--num_captions", type=int, default=2, help="单张图像生成的描述数量")
    parser.add_argument("--contrast_prob", type=float, default=0.5, help="生成同科区别性描述的触发概率")
    parser.add_argument("--simple_prob", type=float, default=0.2, help="生成极简学名模板的概率")
    parser.add_argument("--min_dims", type=int, default=3, help="采样解剖维度的最小数量")
    parser.add_argument("--max_dims", type=int, default=4, help="采样解剖维度的最大数量")
    parser.add_argument("--prompt_prefix", type=str, default="A photo of a", help="Prompt 通用前缀")

    parser.add_argument("--confusion_json", default=None, help="可选：由 valid.py 导出的 confusion_intensity.json 路径")
    parser.add_argument("--hard_prob", type=float, default=0.8, help="按混淆强度针对性采样的概率 (其余回退同科)")

    parser.add_argument("--make_lmdb", action="store_true", help="是否打包单文件 LMDB")
    parser.add_argument("--lmdb_out", default=None, help="LMDB 保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    pipeline = DatasetPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
