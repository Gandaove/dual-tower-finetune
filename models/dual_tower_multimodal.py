from __future__ import annotations

import datetime as dt
import json
import pickle
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import peft
import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from models.loss import build_loss
from models.optimizer import build_optimizer, build_scheduler
from utils.config import Config, LoRAConfig as LoraSettingConfig, ModelConfig, TrainConfig
from utils.logger import get_logger

_log = get_logger("model")


def _list_to_tensor(obj: Any) -> torch.Tensor:
    return torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8).clone()


def _build_peft_config(cfg: LoraSettingConfig, targets: List[str]):
    t = getattr(cfg, "type", "lora").lower()
    if t in ("lora", "dora"):
        return peft.LoraConfig(
            r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
            bias=cfg.bias, target_modules=targets, use_dora=(t == "dora"),
        )
    cls_map = {"loha": peft.LoHaConfig, "lokr": peft.LoKrConfig}
    if t in cls_map:
        return cls_map[t](r=cfg.r, alpha=cfg.alpha, module_dropout=cfg.dropout, target_modules=targets)
    raise ValueError(f"不支持的 LoRA 类型: {t}")


class BaseDualTowerModel(LightningModule, ABC):
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        classes: Optional[List[str]] = None,
        transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.classes = classes or []
        self.transform = transform
        self.arch = model_cfg.arch.lower()
        self.max_text_length = model_cfg.max_text_length if model_cfg else 64

        if model_cfg.compile and model_cfg.use_lora:
            raise ValueError("配置冲突: compile=True 与 use_lora=True 无法同时开启。")

        self.model, self.processor = self._build_backbone()

        self.customize_text_length(self.max_text_length)

        self._configure_freezing()
        self._apply_dropout_trunk(self.model, model_cfg.dropout)
        if model_cfg.compile:
            self._setup_compile()

        self.loss_fn = build_loss(model_cfg)
        if hasattr(self.loss_fn, "sync_from_model_params"):
            self.loss_fn.sync_from_model_params(self.model)

        self._init_ema()
        self._optimizer = None
        self._scheduler = None
        self._sched_interval = "step"
        self._opt_extra = None
        self._log_total, self._log_train_loss = 0.0, 0.0
        self.padding_func = self.model_cfg.text_padding
        if self.arch == "fgclip2" or self.model_cfg.compile:
            self.padding_func = "max_length"

    @property
    def ckpt_suffix(self) -> str:
        """根据微调模式动态派生保存文件名后缀"""
        return "-lora" if self.model_cfg.use_lora else ""

    @abstractmethod
    def get_vision_tower(self) -> nn.Module:
        pass

    @abstractmethod
    def set_vision_tower(self, module: nn.Module) -> None:
        pass

    @abstractmethod
    def get_text_tower(self) -> nn.Module:
        pass

    @abstractmethod
    def set_text_tower(self, module: nn.Module) -> None:
        pass

    @property
    def vision_tower(self) -> nn.Module:
        return self.get_vision_tower()

    @property
    def text_tower(self) -> nn.Module:
        return self.get_text_tower()

    @property
    def raw_text_tower(self) -> nn.Module:
        """剥离 LoRA/PEFT 包装，直接获取底层原生文本塔对象"""
        tt = self.text_tower
        if hasattr(tt, "base_model"):
            tt = getattr(tt.base_model, "model", tt.base_model)
        return tt

    @abstractmethod
    def _build_backbone(self) -> Tuple[nn.Module, Any]:
        """加载各自权重并完成专用属性修复"""
        pass

    @abstractmethod
    def get_default_text_length(self) -> int:
        """获取模型原生的最大文本长度"""
        pass

    @abstractmethod
    def _get_token_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        """提供待冻结的 Token Embedding 模块/参数"""
        pass

    @abstractmethod
    def _get_position_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        """提供待插值的位置编码模块/参数"""
        pass

    @abstractmethod
    def _update_text_config_length(self, target_len: int):
        """同步更新具体模型内部 config 中的长度定义"""
        pass

    @staticmethod
    def _interpolate_weight(weight_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """对权重张量在序列长度维度进行 1D 线性插值"""
        old_len = weight_tensor.shape[0]
        w_3d = weight_tensor.unsqueeze(0).permute(0, 2, 1)  # [1, Dim, old_len]
        new_w = F.interpolate(
            w_3d, size=target_len, mode="linear", align_corners=True
        ).permute(0, 2, 1).squeeze(0).contiguous()          # [target_len, Dim]
        return new_w

    def _interpolate_item(self, item: Union[nn.Module, nn.Parameter], target_len: int):
        """执行原位替换，避免破坏模块引用结构。"""
        if isinstance(item, nn.Embedding):
            if target_len <= item.weight.shape[0]:
                return
            new_w = self._interpolate_weight(item.weight.data, target_len)
            item.weight = nn.Parameter(new_w.to(device=item.weight.device, dtype=item.weight.dtype))
            item.num_embeddings = target_len

        elif isinstance(item, nn.Parameter):
            if target_len <= item.shape[0]:
                return
            new_w = self._interpolate_weight(item.data, target_len)
            item.data = new_w.to(device=item.device, dtype=item.dtype)

        elif isinstance(item, nn.Module) and hasattr(item, "weight"):
            if target_len <= item.weight.shape[0]:
                return
            new_w = self._interpolate_weight(item.weight.data, target_len)
            item.weight = nn.Parameter(new_w.to(device=item.weight.device, dtype=item.weight.dtype))

    def _freeze_token_embedding(self):
        '''冻结 Token Embeddings'''
        token_embs = self._get_token_embeddings()
        for te in token_embs:
            if isinstance(te, nn.Module):
                for p in te.parameters():
                    p.requires_grad = False
            elif isinstance(te, nn.Parameter):
                te.requires_grad = False
        _log.info(f"[{self.arch.upper()}] 已冻结 Token Embedding 参数")

    def _sync_tokenizer_max_length(self, target_len: int):
        """(for TIPsv2 only)同步更新 processor 的 model_max_length，防止tokenizer静默截断"""
        if self.processor is None:
            return

        tok = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tok, "model_max_length"):
            tok.model_max_length = target_len
            _log.info(f"[{self.arch.upper()}] 同步更新 Tokenizer.model_max_length = {target_len}")

    def customize_text_length(self, target_length: int):
        """仅在目标长度超过模型原生基准时，激活插值与 Token Embedding 冻结。"""
        default_len = self.get_default_text_length()
        if target_length <= default_len:
            self.is_text_extended = False
            return
        
        self.is_text_extended = True
        _log.info(
            f"[{self.arch.upper()}] 检测到最大文本长度设置变大: 设置 {target_length} > 基准 {default_len}。"
            "正在执行 1D 位置编码插值，并冻结 Token Embedding..."
        )

        for pos_item in self._get_position_embeddings():
            self._interpolate_item(pos_item, target_length)

        self._update_text_config_length(target_length)
        if self.arch == "tipsv2":
            self._sync_tokenizer_max_length(target_length)
            

    @abstractmethod
    def encode_vision(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """实现前向图像抽取，返回 (pooler_output, last_hidden_state)"""
        pass

    @abstractmethod
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """实现前向文本抽取，返回 (pooler_output, last_hidden_state)"""
        pass

    def _configure_freezing(self):
        vt, tt = self.vision_tower, self.text_tower
        fm = self.train_cfg.freeze_mode
        lora = self.model_cfg.lora
        use_lora = self.model_cfg.use_lora

        text_is_trainable = fm not in ("both", "text")
        # 仅当实际触发了文本最大长度扩展、且text_tower全量微调时，拦截操作
        if self.is_text_extended and text_is_trainable:
            if not use_lora or "text" not in lora.apply_to:
                raise ValueError(
                    f"安全策略拦截: 禁止在 max_text_length={self.max_text_length} 超过了 "
                    f"[{self.arch.upper()}] 的原始设置时于text端采用全量微调。\n"
                    "在此扩展适配模式下，触发了自定义位置编码插值；为防止语言表征崩溃，文本塔禁止全量微调，暂时仅允许通过 LoRA 微调！\n"
                    "请在 model_config 中设置 use_lora=True 且 lora.apply_to 包含 'text'；"
                    "或将 freeze_mode 设置为 'text' / 'both' 以确保文本端冻结。"
                )

        if fm in ("both", "vision") and vt: vt.requires_grad_(False)
        elif vt: vt.requires_grad_(True)

        if fm in ("both", "text") and tt: tt.requires_grad_(False)
        elif tt: tt.requires_grad_(True)

        if use_lora:
            if "vision" in lora.apply_to and vt:
                peft_vt = peft.get_peft_model(
                    vt, _build_peft_config(lora, lora.target_modules)
                )
                self.set_vision_tower(peft_vt)
            if "text" in lora.apply_to and tt:
                peft_tt = peft.get_peft_model(
                    tt, _build_peft_config(lora, lora.target_modules)
                )
                self.set_text_tower(peft_tt)

        if self.is_text_extended:
            self._freeze_token_embedding()
            # 显式解冻位置编码
            for pos_item in self._get_position_embeddings():
                if isinstance(pos_item, nn.Module):
                    for p in pos_item.parameters():
                        p.requires_grad = True
                elif isinstance(pos_item, nn.Parameter):
                    pos_item.requires_grad = True
            _log.info(f"[{self.arch.upper()}] 已激活 Position Embedding 进行微调")

        if fm in ("vision", "both") and ("vision" not in lora.apply_to or not use_lora) and self.vision_tower:
            self.vision_tower.eval()
        if fm in ("text", "both") and ("text" not in lora.apply_to or not use_lora) and self.text_tower:
            self.text_tower.eval()

    def _apply_dropout_trunk(self, model: nn.Module, p: float):
        if not p or p <= 0: return
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def _setup_compile(self):
        if self.train_cfg.freeze_mode not in ("vision", "both") and self.vision_tower:
            self.set_vision_tower(torch.compile(self.vision_tower, dynamic=True))
        if self.train_cfg.freeze_mode not in ("text", "both") and self.text_tower:
            self.set_text_tower(torch.compile(self.text_tower, dynamic=True))

    def tokenize_text(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str): texts = [texts]
        if self.processor is not None:
            encoded = self.processor(
                text=texts, return_tensors="pt", truncation=True,
                padding=self.padding_func,
                max_length=self.max_text_length,
            )
            return dict(encoded)
        raise RuntimeError("未找到可用的 Processor")

    def get_vision_embedding(self, pixel_values: torch.Tensor, to_float32: bool = True, return_dense: bool = False):
        feats, patch_feats = self.encode_vision(pixel_values)
        if to_float32:
            feats = feats.float()
            patch_feats = patch_feats.float() if patch_feats is not None else None
        return (feats, patch_feats) if return_dense else feats

    def get_text_embedding(self, text_inputs: Dict[str, torch.Tensor], to_float32: bool = True, return_dense: bool = False):
        feats, word_feats = self.encode_text(text_inputs)
        if to_float32:
            feats = feats.float()
            word_feats = word_feats.float() if word_feats is not None else None
        return (feats, word_feats) if return_dense else feats

    def get_embeddings(
        self, images: Optional[Dict[str, torch.Tensor]] = None,
        texts: Optional[Dict[str, torch.Tensor]] = None,
        to_float32: bool = True, return_dense: bool = False,
    ):
        res = []
        if images is not None:
            res.append(self.get_vision_embedding(images.get("pixel_values", images.get("image")), to_float32, return_dense))
        if texts is not None:
            res.append(self.get_text_embedding(texts, to_float32, return_dense))

        if len(res) == 2:
            return (res[0][0], res[1][0], res[0][1], res[1][1]) if return_dense else tuple(res)
        return res[0] if res else None

    # ---------------- EMA / 训练 / 验证 ---------------- #
    def _init_ema(self):
        self.ema_decay = self.model_cfg.ema if (self.model_cfg.ema and 0 < self.model_cfg.ema < 1.0) else None
        if not self.ema_decay:
            self.ema_shadow = None
            return
        self.ema_shadow = {
            name: p.detach().clone() for name, p in self.named_parameters() if p.requires_grad and p.is_floating_point()
        }

    @torch.no_grad()
    def update_ema(self):
        if self.ema_shadow is None: return
        for name, p in self.named_parameters():
            if name not in self.ema_shadow: continue
            sp = self.ema_shadow[name]
            if sp.device != p.device: sp = sp.to(p.device); self.ema_shadow[name] = sp
            sp.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    @contextmanager
    def use_ema(self):
        if self.ema_shadow is None:
            yield
            return
        backup = {name: p.data for name, p in self.named_parameters() if name in self.ema_shadow}
        for name in backup: self.get_parameter(name).data = self.ema_shadow[name]
        try: yield
        finally:
            for name, data in backup.items(): self.get_parameter(name).data = data

    @abstractmethod
    def get_num_blocks(self) -> Tuple[int, int]:
        """返回 (n_vision_blocks, n_text_blocks)"""
        pass

    def configure_optimizers(self):
        n_v_blocks, n_t_blocks = self.get_num_blocks()

        opt, extra = build_optimizer(
            self, 
            self.model_cfg, 
            n_v_blocks=n_v_blocks, 
            n_t_blocks=n_t_blocks
        )
        self._optimizer, self._opt_extra = opt, extra
        sched, interval = build_scheduler(
            opt, 
            self.model_cfg, 
            self.train_cfg.epochs, 
            getattr(self, "_steps_per_epoch", 2000), 
            self.model_cfg.scheduler.interval
        )
        self._scheduler, self._sched_interval = sched, interval
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": interval}}

    def base_step(self, batch):
        images = batch["image"]
        text_inputs = {k: v.to(self.device) for k, v in self.tokenize_text(batch["text"]).items()}
        img_emb = self.get_vision_embedding(images, to_float32=False, return_dense=False)
        txt_emb = self.get_text_embedding(text_inputs, to_float32=False, return_dense=False)
        return self.loss_fn(img_emb, txt_emb, labels=batch.get("label", None))

    def training_step(self, batch, batch_idx):
        loss = self.base_step(batch)
        self._log_total += 1
        self._log_train_loss += loss.detach().float().item()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, batch_size=batch["image"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.base_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=batch["image"].size(0))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_ema()

    def pack_checkpoint(self, epoch: int, metric: float, save_optimizer: bool, train_config: dict):
        """区分仅保存 LoRA 可训参数 / 全量模型"""
        is_lora = self.model_cfg.use_lora
        if is_lora:
            trainable_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
            sd = {
                k.replace("_orig_mod.", ""): v.cpu()
                for k, v in self.model.state_dict().items()
                if k.replace("_orig_mod.", "") in trainable_names
            }
        else:
            sd = {k.replace("_orig_mod.", ""): v.cpu() for k, v in self.model.state_dict().items()}

        sd.update({f"loss_fn.{k}": v.cpu() for k, v in self.loss_fn.state_dict().items()})

        meta = {
            "model": "dual_tower", "arch": self.arch, "epoch": str(epoch),
            "save_metric": str(float(metric)), "time": dt.datetime.now().isoformat(),
            "use_lora": str(is_lora),
            "max_text_length": str(self.max_text_length),
            "optimizer": "present" if save_optimizer else "none",
            "scheduler": "present" if save_optimizer else "none",
            "train_config": json.dumps(train_config, ensure_ascii=False),
        }

        if not save_optimizer or self._optimizer is None: return sd, meta

        opt_sd = self._optimizer.state_dict()
        for k, v in opt_sd.get("state", {}).items():
            sd.update({f"optimizer.state.{k}.{sk}": sv.cpu().contiguous() for sk, sv in v.items() if isinstance(sv, torch.Tensor)})

        pgs = [{k: v for k, v in pg.items() if k != "params"} for pg in opt_sd.get("param_groups", [])]
        sd["optimizer.param_groups"] = _list_to_tensor(pgs)

        if self._scheduler is not None:
            for k, v in self._scheduler.state_dict().items():
                if isinstance(v, torch.Tensor): sd[f"scheduler.{k}"] = v.cpu().contiguous()
                else: meta[f"scheduler.{k}"] = json.dumps(v)
        return sd, meta

    def save_safetensors(self, path: str, epoch: int, metric: float, save_optimizer: bool, train_config: dict):
        tensors, meta = self.pack_checkpoint(epoch, metric, save_optimizer, train_config)
        st.save_file(tensors, path, metadata=meta)

    def load_safetensors(self, path: str, load_opt: bool = True):
        with st.safe_open(path, framework="pt") as f:
            meta, keys = f.metadata() or {}, f.keys()
            # self.model.load_state_dict({k: f.get_tensor(k) for k in keys if not k.startswith(("optimizer.", "scheduler.", "loss_fn."))}, strict=False)
            # self.loss_fn.load_state_dict({k[8:]: f.get_tensor(k) for k in keys if k.startswith("loss_fn.")}, strict=False)

            model_state = self.model.state_dict()
            clean_sd = {}
            for k in keys:
                if k.startswith(("optimizer.", "scheduler.", "loss_fn.")):
                    continue
                tensor = f.get_tensor(k)
                if k in model_state:
                    target_param = model_state[k]
                    # 拦截 [] 与 [1] 的单元素尺寸不匹配
                    if tensor.numel() == target_param.numel() and tensor.shape != target_param.shape:
                        tensor = tensor.view_as(target_param)
                clean_sd[k] = tensor

            self.model.load_state_dict(clean_sd, strict=False)

            loss_state = self.loss_fn.state_dict()
            loss_sd = {}
            for k in keys:
                target_k = None
                # 兼容训练检查点 (带前缀) 与 合并后权重 (裸键名)
                if k.startswith("loss_fn."):
                    target_k = k[8:]
                elif k in ("logit_scale", "logit_bias"):
                    target_k = k

                if target_k is not None:
                    # 在 loss_state 中查找匹配项 (支持直接匹配 logit_scale 或前缀嵌套的 global_loss.logit_scale)
                    for state_k in loss_state.keys():
                        if state_k == target_k or state_k.endswith(f".{target_k}"):
                            tensor = f.get_tensor(k)
                            target_param = loss_state[state_k]
                            if tensor.numel() == target_param.numel() and tensor.shape != target_param.shape:
                                tensor = tensor.view_as(target_param)
                            loss_sd[state_k] = tensor

            if loss_sd:
                self.loss_fn.load_state_dict(loss_sd, strict=False)

            # 若 checkpoint 内无独立 loss 标量，被动同步主干参数
            if hasattr(self.loss_fn, "sync_from_model_params"):
                self.loss_fn.sync_from_model_params(self.model)

            if not (load_opt and self._optimizer):
                return meta
            state = {}
            for k in (k for k in keys if k.startswith("optimizer.state.")):
                pid_str, sk = k[16:].split(".", 1)
                state.setdefault(int(pid_str), {})[sk] = f.get_tensor(k)

            cur_pgs = self._optimizer.state_dict()["param_groups"]
            if "optimizer.param_groups" in keys:
                try:
                    saved_pgs = pickle.loads(bytes(f.get_tensor("optimizer.param_groups").numpy()))
                    for cur_pg, saved_pg in zip(cur_pgs, saved_pgs): cur_pg.update(saved_pg)
                except Exception as e: _log.warning(f"[load_safetensors] param_groups 恢复失败: {e}")

            if state: self._optimizer.load_state_dict({"state": state, "param_groups": cur_pgs})
            if self._scheduler:
                try: self._scheduler.load_state_dict({k[10:]: f.get_tensor(k) for k in keys if k.startswith("scheduler.")})
                except Exception as e: _log.warning(f"[load_safetensors] scheduler 恢复失败: {e}")
        return meta


class Siglip(BaseDualTowerModel):
    def get_vision_tower(self) -> nn.Module: return self.model.vision_model
    def set_vision_tower(self, module: nn.Module): self.model.vision_model = module
    def get_text_tower(self) -> nn.Module: return self.model.text_model
    def set_text_tower(self, module: nn.Module): self.model.text_model = module

    def get_num_blocks(self) -> Tuple[int, int]:
        vt = self.vision_tower.base_model.model if hasattr(self.vision_tower, "base_model") else self.vision_tower
        tt = self.text_tower.base_model.model if hasattr(self.text_tower, "base_model") else self.text_tower
        return len(vt.encoder.layers), len(tt.encoder.layers)

    def _build_backbone(self):
        m = AutoModel.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=self.model_cfg.trust_remote_code)
        try: p = AutoProcessor.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=self.model_cfg.trust_remote_code)
        except Exception: p = AutoTokenizer.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=self.model_cfg.trust_remote_code)
        return m, p

    def get_default_text_length(self) -> int:
        return getattr(self.model.config.text_config, "max_position_embeddings", 64)

    def _get_token_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        return [self.raw_text_tower.embeddings.token_embedding]

    def _get_position_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        return [self.raw_text_tower.embeddings.position_embedding]

    def _update_text_config_length(self, target_len: int):
        self.model.config.text_config.max_position_embeddings = target_len
        emb = self.raw_text_tower.embeddings
        emb.register_buffer(
            "position_ids",
            torch.arange(target_len, device=emb.position_embedding.weight.device).expand((1, -1)),
            persistent=False,
        )

    def encode_vision(self, pixel_values: torch.Tensor):
        vt = self.vision_tower if self.training else getattr(self.vision_tower, "_orig_mod", self.vision_tower)
        out = vt(pixel_values=pixel_values)
        return out.pooler_output, getattr(out, "last_hidden_state", None)

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]):
        tt = self.text_tower if self.training else getattr(self.text_tower, "_orig_mod", self.text_tower)
        out = tt(**text_inputs)
        return out.pooler_output, getattr(out, "last_hidden_state", None)


class TIPsv2(BaseDualTowerModel):
    def get_vision_tower(self) -> nn.Module: return self.model.vision_encoder
    def set_vision_tower(self, module: nn.Module): self.model.vision_encoder = module
    def get_text_tower(self) -> nn.Module: return self.model.text_encoder
    def set_text_tower(self, module: nn.Module): self.model.text_encoder = module

    def get_num_blocks(self) -> Tuple[int, int]:
        vt = self.vision_tower.base_model.model if hasattr(self.vision_tower, "base_model") else self.vision_tower
        tt = self.text_tower.base_model.model if hasattr(self.text_tower, "base_model") else self.text_tower
        return len(vt.blocks), len(tt.transformer.resblocks)

    def _build_backbone(self):
        m = AutoModel.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=True)
        try: p = AutoProcessor.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=True)
        except Exception: p = AutoTokenizer.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=True)
        return m, p

    def get_default_text_length(self) -> int:
        return getattr(self.model.config, "max_len", 64)

    def _get_token_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        return [self.raw_text_tower.token_embedding]

    def _get_position_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        return [self.raw_text_tower.pos_embedder]

    def _update_text_config_length(self, target_len: int):
        self.model.config.max_len = target_len
        if hasattr(self.raw_text_tower, "context_length"):
            self.raw_text_tower.context_length = target_len

    def encode_vision(self, pixel_values: torch.Tensor):
        vt = self.vision_tower if self.training else getattr(self.vision_tower, "_orig_mod", self.vision_tower)
        out = vt(pixel_values)
        return out[0].squeeze(1), out[2]

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]):
        tt = self.text_tower if self.training else getattr(self.text_tower, "_orig_mod", self.text_tower)
        input_ids = text_inputs.get("input_ids", text_inputs.get("ids"))
        padding_mask = text_inputs.get("padding_mask")
        if padding_mask is None and "attention_mask" in text_inputs:
            padding_mask = 1.0 - text_inputs["attention_mask"].float()
        out = tt(input_ids, padding_mask)
        return out, None


class FGClip(BaseDualTowerModel):
    def get_vision_tower(self) -> nn.Module: return self.model.vision_model
    def set_vision_tower(self, module: nn.Module): self.model.vision_model = module
    def get_text_tower(self) -> nn.Module: return self.model.text_model
    def set_text_tower(self, module: nn.Module): self.model.text_model = module

    def get_num_blocks(self) -> Tuple[int, int]:
        vt = self.vision_tower.base_model.model if hasattr(self.vision_tower, "base_model") else self.vision_tower
        tt = self.text_tower.base_model.model if hasattr(self.text_tower, "base_model") else self.text_tower
        return len(vt.encoder.layers), len(tt.encoder.layers)

    def customize_text_length(self, target_length: int):
        """支持任意文本长度配置 (涵盖短文本 64、原生 196 及超长外推插值)"""
        self.max_text_length = target_length
        default_len = self.get_default_text_length()

        if target_length <= 64:
            # 模式 1: 回退至原生短文本体系
            self.is_text_extended = False
            _log.info(f"[FGCLIP2] 配置短文本长度 {target_length} <= 64，启用 Short-Text 编码分支")
            return

        if target_length <= default_len:
            # 模式 2: 落在 [65, 196] 预训练区间，利用原生权重切片，无需插值
            self.is_text_extended = False
            self._update_text_config_length(target_length)
            _log.info(f"[FGCLIP2] 配置长文本长度 {target_length} <= {default_len}，启用 Long-Text 编码分支")
            return

        # 模式 3: 超过 196，激活 1D 线性外推插值
        self.is_text_extended = True
        _log.info(f"[FGCLIP2] 配置超长文本长度 {target_length} > {default_len}，激活双轨位置编码外推插值...")
        for pos_item in self._get_position_embeddings():
            self._interpolate_item(pos_item, target_length)

        self._update_text_config_length(target_length)

    def _build_backbone(self):
        m = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.pretrained_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        self._patch_text_embeddings(m)
        p = AutoProcessor.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=True)
        # p = AutoTokenizer.from_pretrained(self.model_cfg.pretrained_path, trust_remote_code=True)
        return m, p

    def get_default_text_length(self) -> int:
        cfg = getattr(self.model.config, "text_config", self.model.config)
        return getattr(cfg, "longtext_len", 196)

    def _get_token_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        tt = self.raw_text_tower
        return [tt.embeddings.token_embedding]

    def _get_position_embeddings(self) -> List[Union[nn.Module, nn.Parameter]]:
        emb = self.raw_text_tower.embeddings
        items = []
        if self.max_text_length <= 64:
            items.append(emb.position_embedding)
        else:
            for extra in ("position_embedding_res", "position_embedding_ori"):
                if hasattr(emb, extra):
                    items.append(getattr(emb, extra))
        return items

    def _update_text_config_length(self, target_len: int):
        cfg = getattr(self.model.config, "text_config", self.model.config)
        cfg.longtext_len = target_len
        keep_len = getattr(cfg, "keep_len", 64)

        emb = self.raw_text_tower.embeddings
        mask1 = torch.zeros([target_len, 1], dtype=torch.float32)
        mask2 = torch.zeros([target_len, 1], dtype=torch.float32)
        mask1[:keep_len, :] = 1.0
        mask2[keep_len:, :] = 1.0
        pos = torch.arange(target_len, dtype=torch.long).unsqueeze(0)

        for name, tensor in (("mask1", mask1), ("mask2", mask2), ("position_ids", pos)):
            if hasattr(emb, name):
                delattr(emb, name)
            emb.register_buffer(name, tensor, persistent=False)

    def _patch_text_embeddings(self, model: nn.Module):
        """解决 FG-CLIP2 源码中 (text)Embedding 的 mask 与 position_ids 未随权重初始化导致的 meta 空置缺陷"""
        tt = getattr(model, "text_model", None)
        emb = getattr(tt, "embeddings", None)
        if emb is None: return
        cfg = getattr(model.config, "text_config", model.config)
        l_len, k_len = getattr(cfg, "longtext_len", 196), getattr(cfg, "keep_len", 64)

        m1, m2 = torch.zeros([l_len, 1], dtype=torch.float32), torch.zeros([l_len, 1], dtype=torch.float32)
        m1[:k_len, :] = 1.0
        m2[k_len:, :] = 1.0
        pos = torch.arange(l_len, dtype=torch.long).unsqueeze(0)

        for name, tensor in (("mask1", m1), ("mask2", m2), ("position_ids", pos)):
            if hasattr(emb, name): delattr(emb, name)
            emb.register_buffer(name, tensor, persistent=False)

    def encode_vision(self, pixel_values: torch.Tensor):
        vt = self.vision_tower if self.training else getattr(self.vision_tower, "_orig_mod", self.vision_tower)
        b, c, h, w = pixel_values.shape
        raw_vt = getattr(vt, "base_model", vt)
        p = getattr(getattr(raw_vt, "config", None), "patch_size", 16)
        h_p, w_p = h // p, w // p

        patches = (
            pixel_values.permute(0, 2, 3, 1).reshape(-1, h_p, p, w_p, p, c)
            .permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, h_p * w_p, p * p * c)
        )

        shapes = torch.tensor([[h_p, w_p]], device=pixel_values.device, dtype=torch.long).expand(b, -1)
        mask = torch.ones((b, h_p * w_p), device=pixel_values.device, dtype=torch.bool)
        out = vt(pixel_values=patches, spatial_shapes=shapes, attention_mask=mask)
        return out.pooler_output, getattr(out, "last_hidden_state", None)

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]):
        tt = self.text_tower if self.training else getattr(self.text_tower, "_orig_mod", self.text_tower)
        walk_type = "long" if self.max_text_length > 64 else "short"
        out = tt(**text_inputs, walk_type=walk_type)
        pooler = out.pooler_output
        if walk_type == "long" and hasattr(self.model, "longtext_head"):
            pooler = self.model.longtext_head(pooler)
        return pooler, getattr(out, "last_hidden_state", None)


_ARCH_MAP = {
    "siglip2": Siglip,
    "tipsv2": TIPsv2,
    "fgclip2": FGClip,
}


def build_dual_tower_model(
    cfg: Config,
    classes: Optional[List[str]] = None,
    transform=None,
) -> BaseDualTowerModel:
    arch = cfg.model.arch.lower()
    if arch not in _ARCH_MAP:
        raise ValueError(f"不支持的架构类型: {arch}，可选: {list(_ARCH_MAP.keys())}")

    cls = _ARCH_MAP[arch]
    return cls(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        classes=classes,
        transform=transform,
    )


build_model = build_dual_tower_model
