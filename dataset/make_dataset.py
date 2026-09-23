from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm


class TaxonomyMapper:
    """解析物种元数据表 (包含 index, pest_name, alias/pest_cname, pest_latin_name)"""

    def __init__(self, taxonomy_csv: str | Path):
        self.csv_path = Path(taxonomy_csv)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"物种映射表不存在: {self.csv_path}")
        self._alias_to_latin: Dict[str, str] = {}
        self._alias_to_name: Dict[str, str] = {}
        self._load()

    def _load(self):
        df = pd.read_csv(self.csv_path)
        # 兼容 alias 与 pest_cname 列名
        alias_col = "alias" if "alias" in df.columns else ("pest_cname" if "pest_cname" in df.columns else None)
        if not alias_col or "pest_latin_name" not in df.columns:
            raise ValueError(f"物种映射表必须包含 ('alias' 或 'pest_cname') 以及 'pest_latin_name' 列")

        for _, row in df.iterrows():
            alias = str(row[alias_col]).strip()
            latin = str(row["pest_latin_name"]).strip()
            pest_name = str(row["pest_name"]).strip() if "pest_name" in df.columns else ""
            if alias and latin and alias.lower() != "nan" and latin.lower() != "nan":
                self._alias_to_latin[alias.lower()] = latin
                if pest_name:
                    self._alias_to_name[alias.lower()] = pest_name

    def get_latin_name(self, alias: str) -> Optional[str]:
        return self._alias_to_latin.get(alias.strip().lower())


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
        positive_latins: Set[str],
        num_dim_range: Tuple[int, int] = (3, 4),
        contrast_prob: float = 0.5,
        simple_prob: float = 0.3,
        prompt_prefix: str = "A photo of a",
    ):
        self.kb_loader = kb_loader
        self.positive_latins = positive_latins
        self.num_dim_range = num_dim_range
        self.contrast_prob = contrast_prob
        self.simple_prob = simple_prob
        self.prompt_prefix = prompt_prefix

    def _select_competitor(self, current_latin: str) -> Optional[str]:
        family = self.kb_loader.get_family(current_latin)
        if not family:
            return None

        all_in_family = set(self.kb_loader.family_to_species.get(family, []))
        all_in_family.discard(current_latin.lower().strip())

        # 优先选择同科正类
        pos_in_family = [s for s in all_in_family if s in self.positive_latins]
        if pos_in_family:
            return random.choice(pos_in_family)

        # 次选同科其他物种
        other_in_family = list(all_in_family)
        if other_in_family:
            return random.choice(other_in_family)
        return None

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

            contrast_phrase = None
            if random.random() < self.contrast_prob:
                competitor = self._select_competitor(latin_name)
                if competitor:
                    contrast_phrase = BaseCaptionFormatter.build_contrast_phrase(info, competitor)

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
        tax_mapper: TaxonomyMapper,
        kb_loader: KnowledgeBaseLoader,
        positive_latins: Set[str],
        num_dim_range: Tuple[int, int] = (2, 4),
        contrast_prob: float = 0.5,
        simple_prob: float = 0.2,
        prompt_prefix: str = "A photo of a",
    ):
        self.tax_mapper = tax_mapper
        self.kb_loader = kb_loader
        self.positive_latins = positive_latins
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
            cleaned = re.sub(r"\d+", "", part).strip()
            if cleaned:
                return cleaned.lower()
        return "negative"

    def _select_positive_competitor_in_family(self, current_latin: str) -> Optional[str]:
        family = self.kb_loader.get_family(current_latin)
        if not family:
            return None

        all_in_family = set(self.kb_loader.family_to_species.get(family, []))
        all_in_family.discard(current_latin.lower().strip())

        # 负样本仅允许挑选同科的正类物种
        pos_in_family = [s for s in all_in_family if s in self.positive_latins]
        if pos_in_family:
            return random.choice(pos_in_family)
        return None

    def process_sample(self, img_name: str, num_captions: int = 2) -> Tuple[str, List[str]]:
        alias = self.extract_alias_from_filename(img_name)
        latin_name = self.tax_mapper.get_latin_name(alias)

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

            contrast_phrase = None
            if random.random() < self.contrast_prob:
                competitor = self._select_positive_competitor_in_family(latin_name)
                if competitor:
                    contrast_phrase = BaseCaptionFormatter.build_contrast_phrase(info, competitor)

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


class DatasetPipeline:
    """顶层管线：协调正负样本分流、物理文件检索与存储产出"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.img_root = Path(args.img_root)
        self.out_dir = Path(args.out_dir)

        self.tax_mapper = TaxonomyMapper(args.taxonomy_csv)
        self.base_loader = BaseDatasetLoader(args.dataset_csv, args.class_map_json)
        self.kb_loader = KnowledgeBaseLoader(args.caption_jsons)

        # 统计正样本实际存在的拉丁名集合
        pos_aliases = self.base_loader.get_positive_aliases()
        self.positive_latins: Set[str] = {
            self.tax_mapper.get_latin_name(a).lower()
            for a in pos_aliases
            if self.tax_mapper.get_latin_name(a)
        }

        self.pos_generator = PositiveCaptionGenerator(
            kb_loader=self.kb_loader,
            positive_latins=self.positive_latins,
            num_dim_range=(args.min_dims, args.max_dims),
            contrast_prob=args.contrast_prob,
            simple_prob=args.simple_prob,
            prompt_prefix=args.prompt_prefix,
        )

        self.neg_handler = NegativeSampleHandler(
            tax_mapper=self.tax_mapper,
            kb_loader=self.kb_loader,
            positive_latins=self.positive_latins,
            num_dim_range=(args.min_dims, args.max_dims),
            contrast_prob=args.contrast_prob,
            simple_prob=args.simple_prob,
            prompt_prefix=args.prompt_prefix,
        )

        self.splitter = DatasetSplitter(ratios=args.ratios, seed=args.seed)
        self.manifest_records: List[Tuple[str, List[str]]] = []
        self.image_records: List[Tuple[str, Path]] = []

    def _locate_image(self, candidate_dirs: List[str], img_name: str) -> Optional[Path]:
        for d in candidate_dirs:
            if d:
                p = self.img_root / d / img_name
                if p.is_file():
                    return p
        p = self.img_root / img_name
        if p.is_file():
            return p
        return None

    def process(self):
        missing_imgs = 0
        pos_count = 0
        neg_count = 0

        for img_name, class_id, base_alias in tqdm(self.base_loader.records, desc="[Processing] Records"):
            if class_id == -1 or base_alias.lower() == "negative":
                actual_alias, captions = self.neg_handler.process_sample(
                    img_name, num_captions=self.args.num_captions
                )
                img_file = self._locate_image([actual_alias, "negative"], img_name)
                neg_count += 1
            else:
                actual_alias = base_alias
                latin = self.tax_mapper.get_latin_name(actual_alias)
                captions = self.pos_generator.generate(
                    actual_alias, latin, num_captions=self.args.num_captions
                )
                img_file = self._locate_image([actual_alias], img_name)
                pos_count += 1

            if img_file is None:
                missing_imgs += 1
                continue

            composite_key = f"{actual_alias}+{img_name}"
            self.manifest_records.append((composite_key, captions))
            if self.args.make_lmdb:
                self.image_records.append((composite_key, img_file))

        print(f"[Summary] 完成构建: 正样本 {pos_count} 条, 难分负样本 {neg_count} 条")
        if missing_imgs > 0:
            print(f"[Warning] 磁盘未命中物理图像数: {missing_imgs}")

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
    parser.add_argument("--make_lmdb", action="store_true", help="是否打包单文件 LMDB")
    parser.add_argument("--lmdb_out", default=None, help="LMDB 保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    pipeline = DatasetPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
