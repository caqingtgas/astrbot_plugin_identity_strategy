# /opt/AstrBot/data/plugins/astrbot_plugin_identity_strategy/main.py
from __future__ import annotations
import json
from typing import Optional, Dict, List, Tuple, Any

from astrbot.api import logger, AstrBotConfig
from astrbot.api.star import Star, Context
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest


def _as_obj(name: str, item: Any) -> Optional[Dict]:
    """将配置项从 str 尝试解析为 dict；不是合法 JSON 就丢弃并告警。"""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
            logger.warning("[identity_guard] %s item is JSON but not object: %r", name, obj)
        except Exception as e:
            logger.warning("[identity_guard] %s item parse failed: %r, err=%s", name, s, e)
    return None


def _as_list_of_obj(name: str, value: Any) -> List[Dict]:
    """把 list 里的每一项都转成 dict（兼容字符串 JSON）。"""
    out: List[Dict] = []
    if not isinstance(value, list):
        return out
    for it in value:
        obj = _as_obj(name, it)
        if obj is not None:
            out.append(obj)
        else:
            # 兼容用户把对象再次转成 JSON 字符串后又被 JSON 转义包了一层的情况
            # e.g. "{\"scope\":\"group\",\"target_ids\":[\"123\"]}"
            if isinstance(it, str):
                try:
                    obj2 = json.loads(json.loads(f'"{it}"'))
                    if isinstance(obj2, dict):
                        out.append(obj2)
                        continue
                except Exception:
                    pass
            logger.warning("[identity_guard] invalid %s item ignored: %r", name, it)
    return out


class IdentityGuard(Star):
    """
    身份识别与应对策略注入（无命令、纯配置）
    - 在 on_llm_request 钩子里修改 req.prompt，实现每次对话的前置/后置注入
    - 通过配置绑定到群聊或私聊，支持特殊ID与普通ID分别注入
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config or AstrBotConfig()
        # 关键：把 list 字段统一解析成对象列表
        self._strategies = _as_list_of_obj("strategies", self.config.get("strategies", []))
        self._bindings = _as_list_of_obj("bindings", self.config.get("bindings", []))
        self._default_key = str(self.config.get("default_strategy_key", "") or "").strip()
        self._enabled = bool(self.config.get("enabled", True))
        self._debug_log = bool(self.config.get("debug_log", False))

        logger.info(
            "[identity_guard] loaded. enabled=%s strategies=%d bindings=%d default=%s",
            self._enabled, len(self._strategies), len(self._bindings), self._default_key or "(none)"
        )

    # ---------- 取平台/会话ID（兼容不同版本字段） ----------
    def _get_platform(self, event: AstrMessageEvent) -> str:
        for attr in ("get_platform_name",):
            if hasattr(event, attr):
                try:
                    v = getattr(event, attr)()  # type: ignore
                    if v:
                        return str(v)
                except Exception:
                    pass
        return str(getattr(event, "platform", "") or "")

    def _get_sender_id(self, event: AstrMessageEvent) -> str:
        for path in [
            ("get_sender_id", None),
            (None, ("message_obj", "sender", "user_id")),
        ]:
            name, chain = path
            if name and hasattr(event, name):
                try:
                    v = getattr(event, name)()
                    if v is not None:
                        return str(v)
                except Exception:
                    pass
            if chain:
                try:
                    obj = event
                    for key in chain:
                        obj = getattr(obj, key)
                    if obj is not None:
                        return str(obj)
                except Exception:
                    pass
        return ""

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        for path in [
            ("get_group_id", None),
            (None, ("message_obj", "group_id")),
        ]:
            name, chain = path
            if name and hasattr(event, name):
                try:
                    v = getattr(event, name)()
                    if v is not None:
                        return str(v)
                except Exception:
                    pass
            if chain:
                try:
                    obj = event
                    for key in chain:
                        obj = getattr(obj, key)
                    if obj is not None:
                        return str(obj)
                except Exception:
                    pass
        return ""

    # ---------- 选择绑定的策略 ----------
    def _match_binding(self, event: AstrMessageEvent) -> str:
        platform = (self._get_platform(event) or "").strip()
        sender_id = (self._get_sender_id(event) or "").strip()
        group_id = (self._get_group_id(event) or "").strip()

        for b in self._bindings:
            try:
                b_platform = (b.get("platform") or "").strip()
                if b_platform and b_platform != platform:
                    continue

                scope = (b.get("scope") or "all").strip().lower()
                ids = [str(x).strip() for x in (b.get("target_ids") or [])]

                if scope == "group" and group_id and group_id in ids:
                    return b.get("strategy_key", "")
                if scope == "private" and sender_id and sender_id in ids:
                    return b.get("strategy_key", "")
                if scope == "all" and not ids:
                    return b.get("strategy_key", "")
            except Exception as e:
                logger.warning("[identity_guard] invalid binding %r: %s", b, e)

        return self._default_key

    # ---------- 找到策略 ----------
    def _find_strategy(self, key: str) -> Optional[Dict]:
        if not key or key == "__empty__":
            return None
        for s in self._strategies:
            try:
                if s.get("key") == key and s.get("enabled", True):
                    return s
            except Exception as e:
                logger.warning("[identity_guard] invalid strategy %r: %s", s, e)
        return None

    # ---------- 产出要注入的文本 ----------
    def _build_injection(self, s: Dict, sender_id: str) -> Tuple[str, str]:
        special_ids = [str(x).strip() for x in (s.get("special_ids") or [])]
        if sender_id and sender_id in special_ids:
            return (s.get("message_for_special") or "").strip(), "special"
        return (s.get("message_for_others") or "").strip(), "general"

    # ---------- 在调用 LLM 前注入 ----------
    @filter.on_llm_request(priority=-10000)
    async def inject_guard(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._enabled:
            return
        try:
            sender_id = self._get_sender_id(event)
            strategy_key = self._match_binding(event)
            strategy = self._find_strategy(strategy_key)
            if not strategy:
                return

            text, which = self._build_injection(strategy, sender_id)
            if not text:
                return

            mode = (strategy.get("mode") or "prepend").strip().lower()
            req.prompt = (req.prompt or "")
            if mode == "append":
                req.prompt = f"{req.prompt}\n{text}"
            else:
                req.prompt = f"{text}\n{req.prompt}"

            if self._debug_log:
                logger.info(
                    "[identity_guard] injected(%s) mode=%s platform=%s group=%s sender=%s key=%s",
                    which, mode, self._get_platform(event), self._get_group_id(event), sender_id, strategy_key
                )
        except Exception as e:
            logger.warning("[identity_guard] inject failed: %s", e)
