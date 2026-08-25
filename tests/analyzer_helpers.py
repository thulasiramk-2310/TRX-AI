"""Isolated analyzer fixture without model calls or persistent memory."""
from collections import OrderedDict
from analyzer import RealityAnalyzer
from config import AppConfig
from conversation_memory import ConversationMemory


def analyzer_stub() -> RealityAnalyzer:
    analyzer = RealityAnalyzer.__new__(RealityAnalyzer)
    analyzer.config = AppConfig(use_local_llm=False, vector_memory_enabled=False)
    analyzer.current_user = "guest"
    analyzer.personality = "friendly"
    analyzer._agent_flags = dict.fromkeys(("debug", "improve", "predict"), True)
    analyzer._response_cache = OrderedDict()
    analyzer._review_response_cache = OrderedDict()
    analyzer._conversation_memory = ConversationMemory()
    return analyzer
