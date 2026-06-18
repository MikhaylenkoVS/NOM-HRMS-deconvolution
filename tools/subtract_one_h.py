import re
def subtract_one_h(brutto: str) -> str:
    """
    Временный костыль: уменьшить число H на 1 в строке формулы.
    Работает только для CHON-формул, где H явно указан.
    Примеры:
      C20H29O2 -> C20H28O2
      C10H14O2N -> C10H13O2N
    Если H нет или H1, оставляем как есть.
    """
    if brutto is None or not isinstance(brutto, str):
        return brutto

    m = re.search(r"H(\d+)", brutto)
    if not m:
        # нет явного Hn — ничего не меняем
        return brutto

    h_count = int(m.group(1))
    if h_count <= 1:
        # H или H1 — не рискуем уходить в H0/H-1
        return brutto

    new_h = f"H{h_count - 1}"
    old_h = f"H{h_count}"
    return brutto.replace(old_h, new_h, 1)