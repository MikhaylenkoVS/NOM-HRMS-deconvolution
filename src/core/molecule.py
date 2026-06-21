from src.core.atoms import Atom
from collections import defaultdict
from typing import List, Tuple, Dict


class Molecule:
    """Class representing a molecule"""

    def __init__(self, formula: str = ""):
        self.formula = formula          # stored label, not parsed into atoms
        self.atoms: List[Atom] = []
        self.edges: List[Tuple[int, int, int]] = []

    def add_atom(self, symbol: str, formal_charge: int = 0) -> int:
        """Add an atom to the molecule. Returns the new atom's index."""
        atom_number = len(self.atoms)
        atom = Atom(symbol, atom_number, formal_charge)
        self.atoms.append(atom)
        return atom_number

    def add_bond(self, atom1: int, atom2: int, bond_order: int = 1) -> None:
        """
        Add a bond between two atoms with full pre‑validation.
        The bond is only added if:
          - both indices are valid and different,
          - the bond does not already exist,
          - each atom has enough remaining valence.
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
        """Check if the molecular graph is fully connected (DFS)."""
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
        """Index of Hydrogen Deficiency (IHD), or degree of unsaturation."""
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
        """Return the molecular formula in Hill notation."""
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
        """
        Generate a SMILES string for the molecule.
        (Proper implementation requires a full canonicalisation algorithm;
        currently not implemented.)
        """
        raise NotImplementedError(
            "Full SMILES generation is not yet implemented. "
            "Use get_formula() for a string representation."
        )

    def __repr__(self) -> str:
        return (f"Molecule({self.get_formula()}, "
                f"{len(self.atoms)} atoms, {len(self.edges)} bonds)")


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a molecular formula string into {element: count}.

    Example: 'C7H6O2' -> {'C': 7, 'H': 6, 'O': 2}
    """
    import re
    elems = defaultdict(int)
    for m in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        el = m.group(1)
        n = int(m.group(2) or '1')
        elems[el] += n
    return dict(elems)


def calculate_IHD(formula: Dict[str, int]) -> float:
    """Calculate Index of Hydrogen Deficiency (IHD) from a formula dict.

    IHD = (2*C + 2 + N - H - X) / 2
    where X = total halogens (F, Cl, Br, I)
    """
    C = formula.get('C', 0)
    H = formula.get('H', 0)
    N = formula.get('N', 0)
    X = sum(formula.get(hal, 0) for hal in ['F', 'Cl', 'Br', 'I'])
    return (2 * C + 2 + N - H - X) / 2


def add_formula(base: Dict[str, int], delta: Dict[str, int], k: int = 1) -> None:
    """Add delta multiplied by k to base (in-place)."""
    for elem, count in delta.items():
        base[elem] = base.get(elem, 0) + count * k