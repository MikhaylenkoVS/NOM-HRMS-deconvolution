from src.core.atoms import Atom
from collections import defaultdict
from typing import List, Tuple, Dict


class Molecule:
    """A molecular graph of atoms and bonds.

    Stores atoms and their connectivity, enforcing valence limits when bonds
    are added. Provides derived chemical descriptors such as the molecular
    formula (Hill notation) and the index of hydrogen deficiency (IHD).

    Parameters
    ----------
    formula : str, optional
        Human-readable formula label. Stored as-is; it is not parsed into
        atoms. Default is an empty string.

    Attributes
    ----------
    atoms : list of Atom
        Atoms of the molecule, indexed by insertion order.
    edges : list of tuple of (int, int, int)
        Bonds as ``(atom1_index, atom2_index, bond_order)`` triples.
    """

    def __init__(self, formula: str = ""):
        self.formula = formula          # stored label, not parsed into atoms
        self.atoms: List[Atom] = []
        self.edges: List[Tuple[int, int, int]] = []

    def add_atom(self, symbol: str, formal_charge: int = 0) -> int:
        """Add an atom to the molecule.

        Parameters
        ----------
        symbol : str
            Chemical element symbol of the new atom.
        formal_charge : int, optional
            Formal charge on the atom. Default is 0.

        Returns
        -------
        int
            Index of the newly created atom.
        """
        atom_number = len(self.atoms)
        atom = Atom(symbol, atom_number, formal_charge)
        self.atoms.append(atom)
        return atom_number

    def add_bond(self, atom1: int, atom2: int, bond_order: int = 1) -> None:
        """Add a bond between two atoms with full pre-validation.

        Parameters
        ----------
        atom1, atom2 : int
            Indices of the two atoms to connect.
        bond_order : int, optional
            Bond order (1 = single, 2 = double, 3 = triple). Default is 1.

        Notes
        -----
        The bond is added only if all of the following hold: both indices are
        in range and distinct, the bond does not already exist, and each atom
        has enough remaining valence. Otherwise the call is a silent no-op,
        leaving the molecule unchanged.
        """
        # Индексы должны быть в допустимом диапазоне
        if atom1 >= len(self.atoms) or atom2 >= len(self.atoms):
            return
        if atom1 == atom2:
            return                     # самосвязывание запрещено

        a1 = self.atoms[atom1]
        a2 = self.atoms[atom2]

        # Защита от дублирования связи
        if atom2 in a1.connections or atom1 in a2.connections:
            return

        # Проверка валентности
        if a1.used_valence + bond_order > a1.valence:
            return
        if a2.used_valence + bond_order > a2.valence:
            return

        # Все проверки пройдены – безопасно добавляем связь.
        # Вызовы add_bond атомов гарантированно вернут True, т.к. мы уже всё проверили.
        a1.add_bond(atom2, bond_order)
        a2.add_bond(atom1, bond_order)
        self.edges.append((atom1, atom2, bond_order))

    def is_connected(self) -> bool:
        """Test whether the molecular graph is a single connected component.

        Returns
        -------
        bool
            ``True`` if every atom is reachable from atom 0 by a
            depth-first traversal (an empty molecule counts as connected),
            ``False`` otherwise.
        """
        if not self.atoms:
            return True

        visited = set()
        stack = [0]

        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            stack.extend(self.atoms[idx].connections)

        return len(visited) == len(self.atoms)

    def calculate_IHD(self) -> float:
        """Compute the index of hydrogen deficiency (IHD) of the molecule.

        Returns
        -------
        float
            The IHD (degree of unsaturation), clamped to be non-negative.

        Notes
        -----
        IHD (also called DBE, double-bond equivalent) counts rings plus
        pi-bonds and is computed as ``(2*C + 2 - H + N - X) / 2``, where
        ``X`` is the total number of halogen atoms (F, Cl, Br, I).
        """
        element_count = defaultdict(int)
        for atom in self.atoms:
            element_count[atom.symbol] += 1

        C = element_count.get('C', 0)
        H = element_count.get('H', 0)
        N = element_count.get('N', 0)
        X = (element_count.get('F', 0) + element_count.get('Cl', 0) +
             element_count.get('Br', 0) + element_count.get('I', 0))

        IHD = (2 * C + 2 - H + N - X) / 2
        return max(0.0, IHD)

    def get_formula(self) -> str:
        """Return the molecular formula in Hill notation.

        Returns
        -------
        str
            Formula string with carbon first, hydrogen second, and the
            remaining elements in alphabetical order (e.g. ``"C7H6O2"``).
            If no carbon is present, all elements are alphabetical.
        """
        element_count = defaultdict(int)
        for atom in self.atoms:
            element_count[atom.symbol] += 1

        if 'C' in element_count:
            order = ['C', 'H'] + sorted(
                el for el in element_count if el not in ('C', 'H')
            )
        else:
            order = sorted(element_count.keys())

        parts = []
        for el in order:
            count = element_count[el]
            parts.append(el if count == 1 else f"{el}{count}")
        return "".join(parts)

    def to_smiles(self) -> str:
        """Generate a SMILES string for the molecule.

        Returns
        -------
        str
            The SMILES representation (not yet available).

        Raises
        ------
        NotImplementedError
            Always. A full canonicalization algorithm is required; use
            :meth:`get_formula` for a string representation instead.
        """
        raise NotImplementedError(
            "Full SMILES generation is not yet implemented. "
            "Use get_formula() for a string representation."
        )

    def __repr__(self) -> str:
        return (f"Molecule({self.get_formula()}, "
                f"{len(self.atoms)} atoms, {len(self.edges)} bonds)")


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a molecular formula string into element counts.

    Parameters
    ----------
    formula : str
        Molecular formula, e.g. ``"C7H6O2"``. Element symbols may be one
        upper-case letter optionally followed by a lower-case letter;
        an omitted count is treated as 1.

    Returns
    -------
    dict of {str: int}
        Mapping of element symbol to atom count,
        e.g. ``{'C': 7, 'H': 6, 'O': 2}``.
    """
    import re
    elems = defaultdict(int)
    for m in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        el = m.group(1)
        n = int(m.group(2) or '1')
        elems[el] += n
    return dict(elems)


def calculate_IHD(formula: Dict[str, int]) -> float:
    """Compute the index of hydrogen deficiency (IHD) from a formula dict.

    Parameters
    ----------
    formula : dict of {str: int}
        Element counts, e.g. ``{'C': 7, 'H': 6, 'O': 2}``.

    Returns
    -------
    float
        The IHD (degree of unsaturation), computed as
        ``(2*C + 2 + N - H - X) / 2`` where ``X`` is the total number of
        halogen atoms (F, Cl, Br, I). Not clamped to be non-negative.
    """
    C = formula.get('C', 0)
    H = formula.get('H', 0)
    N = formula.get('N', 0)
    X = sum(formula.get(hal, 0) for hal in ['F', 'Cl', 'Br', 'I'])
    return (2 * C + 2 + N - H - X) / 2


def add_formula(base: Dict[str, int], delta: Dict[str, int], k: int = 1) -> None:
    """Add a scaled formula into another formula, in place.

    Parameters
    ----------
    base : dict of {str: int}
        Formula to be modified in place; receives the added counts.
    delta : dict of {str: int}
        Formula whose element counts are added to ``base``.
    k : int, optional
        Integer multiplier applied to ``delta`` before adding
        (e.g. number of derivatization groups). Default is 1.

    Returns
    -------
    None
        ``base`` is modified in place.
    """
    for elem, count in delta.items():
        base[elem] = base.get(elem, 0) + count * k