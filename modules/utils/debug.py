def add_debug_info(state, key: str, value):
    """
    Safe helper to attach debug info into state.debug_info
    """
    try:
        if hasattr(state, "debug_info") and isinstance(state.debug_info, dict):
            state.debug_info[key] = value
    except Exception:
        pass
