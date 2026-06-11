from enum import Enum
from typing import List, Dict

ELEMENT_DATA = {
    'H': {'atomic_number': 1, 'valence': 1},
    'C': {'atomic_number': 6, 'valence': 4},
    'N': {'atomic_number': 7, 'valence': 3, 'valence_charged': {-1: 2, 0: 3, 1: 4}},
    'O': {'atomic_number': 8, 'valence': 2, 'valence_charged': {-1: 1, 0: 2, 1: 3}},
    'F': {'atomic_number': 9, 'valence': 1},
    'P': {'atomic_number': 15, 'valence': 3, 'valence_charged': {0: 3, 1: 4}},
    'S': {'atomic_number': 16, 'valence': 2, 'valence_charged': {0: 2, 1: 3}},
    'Cl': {'atomic_number': 17, 'valence': 1},
    'Br': {'atomic_number': 35, 'valence': 1},
    'I': {'atomic_number': 53, 'valence': 1},
}

def calculate_formal_charge(symbol: str, valence_electrons: int,
                           bonds: int, lone_pairs: int) -> int:
    """Вычисление формального заряда атома"""
    return valence_electrons - bonds - 2 * lone_pairs

class Hybridization(Enum):
    """Типы гибридизации атома"""
    SP3 = 'sp3'
    SP2 = 'sp2'
    SP = 'sp'
    UNKNOWN = 'unknown'

class Atom:
    """Класс для представления атома в молекуле"""

    def __init__(self, symbol: str, number: int, formal_charge: int = 0):
        if symbol not in ELEMENT_DATA:
            raise ValueError(f"Неподдерживаемый элемент: {symbol}")

        self.symbol = symbol
        self.number = number
        self.formal_charge = formal_charge

        element = ELEMENT_DATA[symbol]
        self.atomic_number = element['atomic_number']

        if 'valence_charged' in element and formal_charge in element['valence_charged']:
            self.valence = element['valence_charged'][formal_charge]
        else:
            self.valence = element['valence']

        self.connections: List[int] = []
        self.bond_orders: List[int] = []
        self.used_valence = 0
        self.hybridization = Hybridization.UNKNOWN
        self.is_aromatic = False

    def add_bond(self, atom_number: int, bond_order: int = 1) -> bool:
        """Добавить связь с другим атомом"""
        if self.used_valence + bond_order > self.valence:
            return False

        self.connections.append(atom_number)
        self.bond_orders.append(bond_order)
        self.used_valence += bond_order
        self._update_hybridization()
        return True

    def _update_hybridization(self):
        """Автоматическое определение гибридизации"""
        if self.symbol != 'C':
            return

        double_bonds = self.bond_orders.count(2)
        triple_bonds = self.bond_orders.count(3)

        if triple_bonds > 0:
            self.hybridization = Hybridization.SP
        elif double_bonds > 0 or self.is_aromatic:
            self.hybridization = Hybridization.SP2
        else:
            self.hybridization = Hybridization.SP3

    def get_bond_order_to(self, atom_number: int) -> int:
        """Получить порядок связи с атомом"""
        try:
            idx = self.connections.index(atom_number)
            return self.bond_orders[idx]
        except ValueError:
            return 0

    def __repr__(self):
        charge_str = f"{self.formal_charge:+d}" if self.formal_charge != 0 else ""
        return f"Atom({self.symbol}{charge_str}, #{self.number}, val={self.used_valence}/{self.valence})"
