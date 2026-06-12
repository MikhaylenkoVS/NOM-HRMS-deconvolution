from collections import defaultdict
from typing import List, Tuple, Dict
from .atoms import Atom
from .fragments import FUNCTIONAL_GROUPS, FRAGMENT_LIBRARY, ALL_FRAGMENTS, MoleculeFragment, create_cooh, create_oh


class Molecule:
    """Класс для представления молекулы"""

    def __init__(self, formula: str = ""):
        self.formula = formula
        self.atoms: List[Atom] = []
        self.edges: List[Tuple[int, int, int]] = []

    def add_atom(self, symbol: str, formal_charge: int = 0) -> int:
        """Добавить атом в молекулу"""
        atom_number = len(self.atoms)
        atom = Atom(symbol, atom_number, formal_charge)
        self.atoms.append(atom)
        return atom_number

    def add_bond(self, atom1: int, atom2: int, bond_order: int = 1):
        """Добавить связь между атомами"""
        if atom1 >= len(self.atoms) or atom2 >= len(self.atoms):
            return

        success1 = self.atoms[atom1].add_bond(atom2, bond_order)
        success2 = self.atoms[atom2].add_bond(atom1, bond_order)

        if success1 and success2:
            self.edges.append((atom1, atom2, bond_order))

    def is_connected(self) -> bool:
        """Проверка связности графа молекулы (DFS)"""
        if not self.atoms:
            return True

        visited = set()
        stack = [0]

        while stack:
            atom_idx = stack.pop()
            if atom_idx in visited:
                continue
            visited.add(atom_idx)
            atom = self.atoms[atom_idx]
            stack.extend(atom.connections)

        return len(visited) == len(self.atoms)

    def calculate_IHD(self) -> int:
        """Индекс водородной недостаточности (IHD)"""
        element_count = defaultdict(int)
        for atom in self.atoms:
            element_count[atom.symbol] += 1

        C = element_count.get('C', 0)
        H = element_count.get('H', 0)
        N = element_count.get('N', 0)
        X = element_count.get('F', 0) + element_count.get('Cl', 0) + \
            element_count.get('Br', 0) + element_count.get('I', 0)

        if H + X < 1:
            return 0

        ihd = (2*C + 2 - H + N - X) / 2
        return max(0, int(ihd))

    def get_formula(self) -> str:
        """Получить молекулярную формулу"""
        element_count = defaultdict(int)
        for atom in self.atoms:
            element_count[atom.symbol] += 1

        formula = ""
        for element in ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']:
            if element in element_count:
                count = element_count[element]
                formula += element if count == 1 else f"{element}{count}"

        return formula

    def to_smiles(self) -> str:
        """Упрощенная генерация SMILES"""
        if not self.atoms:
            return ""

        smiles_parts = []
        for atom in self.atoms:
            symbol = atom.symbol
            if atom.formal_charge > 0:
                symbol += f"+{atom.formal_charge}"
            elif atom.formal_charge < 0:
                symbol += str(atom.formal_charge)
            smiles_parts.append(symbol)

        return "(".join(smiles_parts) + ")" * (len(smiles_parts) - 1)

    def __repr__(self):
        return f"Molecule({self.get_formula()}, {len(self.atoms)} atoms, {len(self.edges)} bonds)"


def parse_formula(formula: str) -> Dict[str, int]:
    """Парсит молекулярную формулу в словарь {элемент: количество}.

    Пример: 'C7H6O2' -> {'C': 7, 'H': 6, 'O': 2}
    """
    import re
    elems = defaultdict(int)
    for m in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        el = m.group(1)
        n = int(m.group(2) or '1')
        elems[el] += n
    return dict(elems)

def calculate_ihd(formula: Dict[str, int]) -> float:
    """Вычисляет степень ненасыщенности (IHD) по формуле.

    IHD = (2*C + 2 + N - H - X) / 2
    где C - углероды, N - азоты, H - водороды, X - галогены
    """
    C = formula.get('C', 0)
    H = formula.get('H', 0)
    N = formula.get('N', 0)
    X = sum(formula.get(hal, 0) for hal in ['F', 'Cl', 'Br', 'I'])
    return (2*C + 2 + N - H - X) / 2

def add_formula(base: Dict[str, int], delta: Dict[str, int], k: int = 1):
    """Добавляет формулу delta к base, умножая на коэффициент k."""
    for elem, count in delta.items():
        base[elem] = base.get(elem, 0) + count * k

print('✅ Вспомогательные функции определены')









