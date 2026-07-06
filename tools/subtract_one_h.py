import re
def subtract_one_h(brutto: str) -> str:
    """Decrease the hydrogen count of a brutto formula by one.

    Helper for converting a neutral formula to its ``[M-H]-`` form at the
    string level.

    Parameters
    ----------
    brutto : str
        CHON brutto formula with an explicit hydrogen count, e.g.
        ``"C20H29O2"``.

    Returns
    -------
    str
        Formula with ``H`` reduced by one (e.g. ``"C20H28O2"``). Returned
        unchanged if the input is not a string, has no explicit ``Hn``, or
        already has ``H`` count ``<= 1`` (to avoid ``H0``/``H-1``).
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