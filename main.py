from __future__ import annotations
import json
from typing import Optional, Dict, List, Tuple, Any

from astrbot.api import logger, AstrBotConfig
from astrbot.api.star import Star, Context
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest


def _decode_json_obj_from_str(name: str, s: str, max_rounds: int = 2) -> Optional[Dict]:
    """
    尝试从字符串中解析出 JSON 对象（dict）。
    兼容历史“字符串里套字符串”的配置：
      - 直接是对象字符串: '{"a":1}'
      - 被转义包了一层:   "{\"a\":1}"  或  "\"{\\\"a\\\":1}\""

    解码策略（最多 max_rounds 轮）：
      1) 先直接 json.loads(cur)。若得到 dict -> 返回；若得到 str -> 继续下一轮；
      2) 若 #1 失败（JSONDecodeError），尝试把 cur 当作“被转义的 JSON 字符串”解一次：json.loads(f'"{cur}"')。
         成功后若得到 str 则进入下一轮；若得到 dict 则返回。

    失败则返回 None，并打印告警日志。
    """
    cur = (s or "").strip()
    if not cur:
        return None

    for round_idx in range(1, max_rounds + 1):
        # 情况1：直接尝试 JSON 解码
        try:
            obj = json.loads(cur)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, str):
                # 继续下一轮，处理“字符串内还是 JSON 文本”的情况
                cur = obj.strip()
                continue
            logger.warning(
                "[identity_guard] %s item is JSON but not object (type=%s): %r",
                name, type(obj).__name__, obj
            )
            return None
        except json.JSONDecodeError:
            # 情况2：把当前字符串视为“被转义的 JSON 字符串”，先反转义再进入下一轮
            try:
                unescaped = json.loads(f'"{cur}"')
            except json.JSONDecodeError as e:
                logger.warning(
                    "[identity_guard] %s item nested-parse failed at round=%d: %r (err=%s)",
                    name, round_idx, cur[:200], e
                )
                return None

            cur = (unescaped or "").strip()
            if not cur:
                return None

    logger.warning(
        "[identity_guard] %s item exceeded max nested decode rounds (%d): %r",
        name, max_rounds, s[:200]
    )
    return None


def _as_obj(name: str, item: Any) -> Optional[Dict]:
    """将配置项解析为 dict；支持 dict 或字符串（字符串按上述兼容规则解析）。"""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return _decode_json_obj_from_str(name, item, max_rounds=2)
    logger.warning("[identity_guard] %s item not dict/str: %r", name, item)
    return None


def _as_list_of_obj(name: str, value: Any) -> List[Dict]:
    """
    把 list 里的每一项都转成 dict。
    - 允许每项是 dict 或“包含 JSON 对象的字符串”（含历史的转义层）
    - 解析失败会记录 WARNING 日志，便于发现配置问题
    """
    out: List[Dict] = []
    if not isinstance(value, list):
        return out
    for it in value:
        obj = _as_obj(name, it)
        if obj is not None:
            out.append(obj)
        else:
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

    # ---------- 通用：事件属性获取器（消除 _get_* 重复逻辑） ----------
    def _get_event_attr(
        self,
        event: AstrMessageEvent,
        paths: List[Tuple[Optional[str], Optional[Tuple[str, ...]]]],
    ) -> str:
        """
        通用事件属性获取器：
        - paths: 列表项为 (method_name, attr_chain)
          * method_name: 先检测 hasattr(event, method_name)，若存在则调用并返回非 None 值
          * attr_chain: 依次 getattr(event, key)，取到非 None 值即返回
        - 仅捕获 AttributeError / TypeError，避免压制其他异常
        """
        for method_name, attr_chain in paths:
            # 方法路径
            if method_name and hasattr(event, method_name):
                try:
                    v = getattr(event, method_name)()
                    if v is not None:
                        return str(v)
                except (AttributeError, TypeError):
                    pass
            # 属性链路径
            if attr_chain:
                try:
                    obj: Any = event
                    for key in attr_chain:
                        obj = getattr(obj, key)
                    if obj is not None:
                        return str(obj)
                except (AttributeError, TypeError):
                    pass
        return ""

    # ---------- 取平台/会话ID（兼容不同版本字段） ----------
    def _get_platform(self, event: AstrMessageEvent) -> str:
        return self._get_event_attr(
            event,
            [
                ("get_platform_name", None),
                (None, ("platform",)),  # 兼容旧字段
            ],
        )

    def _get_sender_id(self, event: AstrMessageEvent) -> str:
        return self._get_event_attr(
            event,
            [
                ("get_sender_id", None),
                (None, ("message_obj", "sender", "user_id")),
            ],
        )

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        return self._get_event_attr(
            event,
            [
                ("get_group_id", None),
                (None, ("message_obj", "group_id")),
            ],
        )

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
            except (KeyError, TypeError, AttributeError) as e:
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
            except (KeyError, TypeError, AttributeError) as e:
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
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning("[identity_guard] inject failed: %s", e)
