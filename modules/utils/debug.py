def add_debug_info(state, key, value):
    try:
        state.add_debug(key, value)
    except Exception:
        # best-effort logging
        print(f"[debug] {key} = {value}")