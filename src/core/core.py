from __future__ import annotations

from ctypes.wintypes import BOOLEAN
from enum import Enum
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nomspectra.spectrum import Spectrum
from collections import defaultdict, Counter
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class Hybridization(Enum):
    """Типы гибридизации атома"""
    SP3 = 'sp3'
    SP2 = 'sp2'
    SP = 'sp'
    UNKNOWN = 'unknown'


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



# === ПОЛНАЯ БИБЛИОТЕКА ФРАГМЕНТОВ ===
FRAGMENT_LIBRARY = {
    # === АЦИКЛИЧЕСКИЕ ФРАГМЕНТЫ ===
    # Одинарные связи C-C
    'methylene': {'heavy_formula': {'C': 1}, 'ihd': 0, 'attachment_points': 2, 'description': 'CH2'},
    'ethylene': {'heavy_formula': {'C': 2}, 'ihd': 0, 'attachment_points': 2, 'description': 'CH2-CH2'},
    'propylene': {'heavy_formula': {'C': 3}, 'ihd': 0, 'attachment_points': 2, 'description': 'CH2-CH2-CH2'},
    # Двойные связи C=C
    'alkene': {'heavy_formula': {'C': 2}, 'ihd': 1, 'attachment_points': 2, 'description': 'CH=CH'},
    'propenyl': {'heavy_formula': {'C': 3}, 'ihd': 1, 'attachment_points': 2, 'description': 'CH=CH-CH2'},
    # Тройные связи C≡C
    'alkyne': {'heavy_formula': {'C': 2}, 'ihd': 2, 'attachment_points': 2, 'description': 'C≡C'},
    'propynyl': {'heavy_formula': {'C': 3}, 'ihd': 2, 'attachment_points': 2, 'description': 'C≡C-CH2'},
    # === 5-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
    'cyclopentane': {'heavy_formula': {'C': 5}, 'ihd': 1, 'attachment_points': 5, 'description': 'Циклопентан'},
    'cyclopentene': {'heavy_formula': {'C': 5}, 'ihd': 2, 'attachment_points': 5, 'description': 'Циклопентен'},
    'cyclopentadiene': {'heavy_formula': {'C': 5}, 'ihd': 3, 'attachment_points': 5, 'description': 'Циклопентадиен'},
    # === 6-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
    'cyclohexane': {'heavy_formula': {'C': 6}, 'ihd': 1, 'attachment_points': 6, 'description': 'Циклогексан'},
    'cyclohexene': {'heavy_formula': {'C': 6}, 'ihd': 2, 'attachment_points': 6, 'description': 'Циклогексен'},
    'cyclohexadiene': {'heavy_formula': {'C': 6}, 'ihd': 3, 'attachment_points': 6, 'description': 'Циклогексадиен'},
    'benzene': {'heavy_formula': {'C': 6}, 'ihd': 4, 'attachment_points': 6, 'description': 'Бензол'},
    # === 8-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
    'cyclooctane': {'heavy_formula': {'C': 8}, 'ihd': 1, 'attachment_points': 8, 'description': 'Циклооктан'},
    'cyclooctene': {'heavy_formula': {'C': 8}, 'ihd': 2, 'attachment_points': 8, 'description': 'Циклооктен'},
    'cyclooctadiene': {'heavy_formula': {'C': 8}, 'ihd': 3, 'attachment_points': 8, 'description': 'Циклооктадиен'},
    'cyclooctatriene': {'heavy_formula': {'C': 8}, 'ihd': 4, 'attachment_points': 8, 'description': 'Циклооктатриен'},
    'cyclooctatetraene': {'heavy_formula': {'C': 8}, 'ihd': 5, 'attachment_points': 8, 'description': 'Циклооктатетраен'},
    # === 10-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
    'cyclodecane': {'heavy_formula': {'C': 10}, 'ihd': 1, 'attachment_points': 10, 'description': 'Циклодекан'},
    'cyclodecene': {'heavy_formula': {'C': 10}, 'ihd': 2, 'attachment_points': 10, 'description': 'Циклодецен'},
    'cyclodecadiene': {'heavy_formula': {'C': 10}, 'ihd': 3, 'attachment_points': 10, 'description': 'Циклодекадиен'},
    'cyclodecatriene': {'heavy_formula': {'C': 10}, 'ihd': 4, 'attachment_points': 10, 'description': 'Циклодекатриен'},
    'cyclodecatetraene': {'heavy_formula': {'C': 10}, 'ihd': 5, 'attachment_points': 10, 'description': 'Циклодекатетраен'},
    'cyclodecapentaene': {'heavy_formula': {'C': 10}, 'ihd': 6, 'attachment_points': 10, 'description': 'Циклодекапентаен'},
    # === КОНДЕНСИРОВАННЫЕ СИСТЕМЫ ===
    'naphthalene': {'heavy_formula': {'C': 10}, 'ihd': 7, 'attachment_points': 8, 'description': 'Нафталин'},
    'anthracene': {'heavy_formula': {'C': 14}, 'ihd': 10, 'attachment_points': 10, 'description': 'Антрацен'},
    # === 5-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С АЗОТОМ ===
    'pyrrolidine': {'heavy_formula': {'C': 4, 'N': 1}, 'ihd': 1, 'attachment_points': 5, 'description': 'Пирролидин'},
    'pyrroline': {'heavy_formula': {'C': 4, 'N': 1}, 'ihd': 2, 'attachment_points': 5, 'description': 'Пирролин'},
    'pyrrole': {'heavy_formula': {'C': 4, 'N': 1}, 'ihd': 3, 'attachment_points': 5, 'description': 'Пиррол'},
    'imidazole': {'heavy_formula': {'C': 3, 'N': 2}, 'ihd': 3, 'attachment_points': 5, 'description': 'Имидазол'},
    'pyrazole': {'heavy_formula': {'C': 3, 'N': 2}, 'ihd': 3, 'attachment_points': 5, 'description': 'Пиразол'},
    # === 5-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С КИСЛОРОДОМ ===
    'tetrahydrofuran': {'heavy_formula': {'C': 4, 'O': 1}, 'ihd': 1, 'attachment_points': 4, 'description': 'Тетрагидрофуран'},
    'dihydrofuran': {'heavy_formula': {'C': 4, 'O': 1}, 'ihd': 2, 'attachment_points': 4, 'description': 'Дигидрофуран'},
    'furan': {'heavy_formula': {'C': 4, 'O': 1}, 'ihd': 3, 'attachment_points': 4, 'description': 'Фуран'},
    # === 6-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С АЗОТОМ ===
    'piperidine': {'heavy_formula': {'C': 5, 'N': 1}, 'ihd': 1, 'attachment_points': 6, 'description': 'Пиперидин'},
    'tetrahydropyridine': {'heavy_formula': {'C': 5, 'N': 1}, 'ihd': 2, 'attachment_points': 6, 'description': 'Тетрагидропиридин'},
    'dihydropyridine': {'heavy_formula': {'C': 5, 'N': 1}, 'ihd': 3, 'attachment_points': 6, 'description': 'Дигидропиридин'},
    'pyridine': {'heavy_formula': {'C': 5, 'N': 1}, 'ihd': 4, 'attachment_points': 6, 'description': 'Пиридин'},
    'pyrimidine': {'heavy_formula': {'C': 4, 'N': 2}, 'ihd': 4, 'attachment_points': 6, 'description': 'Пиримидин'},
    'pyrazine': {'heavy_formula': {'C': 4, 'N': 2}, 'ihd': 4, 'attachment_points': 6, 'description': 'Пиразин'},
    'pyridazine': {'heavy_formula': {'C': 4, 'N': 2}, 'ihd': 4, 'attachment_points': 6, 'description': 'Пиридазин'},
    # === 6-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С КИСЛОРОДОМ ===
    'tetrahydropyran': {'heavy_formula': {'C': 5, 'O': 1}, 'ihd': 1, 'attachment_points': 5, 'description': 'Тетрагидропиран'},
    'dihydropyran': {'heavy_formula': {'C': 5, 'O': 1}, 'ihd': 2, 'attachment_points': 5, 'description': 'Дигидропиран'},
    'pyran': {'heavy_formula': {'C': 5, 'O': 1}, 'ihd': 3, 'attachment_points': 5, 'description': 'Пиран'},}
    # === ФУНКЦИОНАЛЬНЫЕ ГРУППЫ ===
FUNCTIONAL_GROUPS = {    'cooh': {'heavy_formula': {'C': 1, 'O': 2}, 'ihd': 1, 'description': 'Карбоксильная группа'},
                         'oh': {'heavy_formula': {'O': 1}, 'ihd': 0, 'description': 'Гидроксильная группа'},
     'cho': {'heavy_formula': {'C': 1, 'O': 1}, 'ihd': 1, 'description': 'Альдегидная группа'},
     'co': {'heavy_formula': {'C': 1, 'O': 1}, 'ihd': 1, 'description': 'Кетонная группа'},
     'coo': {'heavy_formula': {'C': 1, 'O': 2}, 'ihd': 1, 'description': 'Сложноэфирная группа'},
     'o_ether': {'heavy_formula': {'O': 1}, 'ihd': 0, 'description': 'Простая эфирная связь'},
     'nh2': {'heavy_formula': {'N': 1}, 'ihd': 0, 'description': 'Аминогруппа'},
     'nh': {'heavy_formula': {'N': 1}, 'ihd': 0, 'description': 'Вторичная аминогруппа'},
     'n_tertiary': {'heavy_formula': {'N': 1}, 'ihd': 0, 'description': 'Третичная аминогруппа'},
     'no2': {'heavy_formula': {'N': 1, 'O': 2}, 'ihd': 1, 'description': 'Нитрогруппа'},
     'cn': {'heavy_formula': {'C': 1, 'N': 1}, 'ihd': 2, 'description': 'Нитрильная группа'},
     'conh2': {'heavy_formula': {'C': 1, 'O': 1, 'N': 1}, 'ihd': 1, 'description': 'Амидная группа'},
     'sh': {'heavy_formula': {'S': 1}, 'ihd': 0, 'description': 'Тиольная группа'},
     's_sulfide': {'heavy_formula': {'S': 1}, 'ihd': 0, 'description': 'Сульфидная связь'},
     'so2': {'heavy_formula': {'S': 1, 'O': 2}, 'ihd': 0, 'description': 'Сульфонильная группа'},
     'so3h': {'heavy_formula': {'S': 1, 'O': 3}, 'ihd': 0, 'description': 'Сульфоновая кислота'},
     'f': {'heavy_formula': {'F': 1}, 'ihd': 0, 'description': 'Фтор'},
     'cl': {'heavy_formula': {'Cl': 1}, 'ihd': 0, 'description': 'Хлор'},
     'br': {'heavy_formula': {'Br': 1}, 'ihd': 0, 'description': 'Бром'},
     'i': {'heavy_formula': {'I': 1}, 'ihd': 0, 'description': 'Йод'},}


class FragmentLibrary:
    """Расширенная библиотека молекулярных фрагментов CHNO.

    Каждый метод возвращает матрицу смежности (numpy.ndarray) для фрагмента.
    Матрица задается в терминах тяжёлых атомов (C, N, O), без явных водородов.
    Валентность и число H должны достраиваться алгоритмом генератора, как в основной версии.
    """

    # === УЖЕ ИСПОЛЬЗУЕМЫЕ В ОСНОВНОЙ ВЕРСИИ ФРАГМЕНТЫ ===

    @staticmethod
    def get_benzene():
        """Бензол C6 (ароматическое кольцо).

        Атомы: C0–C5, связи: чередующиеся одинарные/двойные.
        """
        return np.array([
            [0, 2, 0, 0, 0, 1],
            [2, 0, 1, 0, 0, 0],
            [0, 1, 0, 2, 0, 0],
            [0, 0, 2, 0, 1, 0],
            [0, 0, 0, 1, 0, 2],
            [1, 0, 0, 0, 2, 0]
        ], dtype=int)

    @staticmethod
    def get_naphthalene():
        """Нафталин C10 (два конденсированных бензольных кольца)."""
        return np.array([
            [0, 2, 0, 0, 0, 1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 2, 0, 0, 0, 1],
            [1, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 2],
            [0, 0, 0, 0, 1, 0, 0, 1, 2, 0]
        ], dtype=int)

    @staticmethod
    def get_anthracene():
        """Антрацен C14 (три конденсированных бензольных кольца, ИСПРАВЛЕНО)."""
        return np.array([
            [0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
        ], dtype=int)

    @staticmethod
    def get_cooh_group():
        """Карбоксильная группа –COOH, без указания внешнего атома C-скелета.

        Атомы: C0, O1(=O), O2(–OH). Внешний C-скелет должен связываться с C0.
        """
        return np.array([
            [0, 2, 1],
            [2, 0, 0],
            [1, 0, 0]
        ], dtype=int)

    @staticmethod
    def get_oh_group():
        """Гидроксильная группа –OH, представлена только атомом O.

        Внешний атом (обычно C) связывается с этим O, водород достраивается отдельно.
        """
        return np.array([[0]], dtype=int)

    # === НОВЫЕ ФРАГМЕНТЫ CHNO ===

    # 1) Простые ациклические углеродные фрагменты (только C, без H)

    @staticmethod
    def get_methylene():
        """Фрагмент –CH2– как вершина C с двумя свободными связями.

        Матрица 1×1 с нулем: связи к цепи задаются снаружи.
        """
        return np.array([[0]], dtype=int)

    @staticmethod
    def get_ethylene_fragment():
        """Фрагмент –CH2–CH2– : два углерода с одинарной связью.
        Валентность каждого C: 4, одинарная связь между ними.
        Остальные связи заполняются H или другими фрагментами.
        """
        return np.array([
            [0, 1],
            [1, 0]
        ], dtype=int)

    @staticmethod
    def get_alkene_fragment():
        """Фрагмент –CH=CH– : два углерода с двойной связью.
        """
        return np.array([
            [0, 2],
            [2, 0]
        ], dtype=int)

    @staticmethod
    def get_alkyne_fragment():
        """Фрагмент –C#C– : два углерода с тройной связью.
        """
        return np.array([
            [0, 3],
            [3, 0]
        ], dtype=int)

    # 2) Карбонильные фрагменты C=O

    @staticmethod
    def get_carbonyl():
        """Фрагмент –C(=O)– : кетонный/альдегидный карбонил без H.

        Атомы: C0, O1. Внешние связи: C0 должен связываться с двумя соседями
        (для кетона) или с одним соседом и H (для альдегида).
        """
        return np.array([
            [0, 2],
            [2, 0]
        ], dtype=int)

    @staticmethod
    def get_ester_core():
        """Фрагмент сложноэфирного ядра –CO–O– (без внешних C).

        Атомы: C0, O1(=O), O2(эфирный). Внешние связи: C0 и O2.
        """
        return np.array([
            [0, 2, 1],
            [2, 0, 0],
            [1, 0, 0]
        ], dtype=int)

    @staticmethod
    def get_ether_oxygen():
        """Эфирный атом кислорода –O– как отдельный узел.

        Все связи к нему (два C) задаются снаружи.
        """
        return np.array([[0]], dtype=int)

    # 3) Аминные и амидные фрагменты

    @staticmethod
    def get_primary_amine_core():
        """Фрагмент –NH2 как узел N (водороды неявные).

        Матрица 1×1, связи к C задаются при сборке.
        """
        return np.array([[0]], dtype=int)

    @staticmethod
    def get_secondary_amine_core():
        """Фрагмент –NH– : N с двумя связями к C и одним H.
        Представлен как одиночный N, связи к C снаружи.
        """
        return np.array([[0]], dtype=int)

    @staticmethod
    def get_tertiary_amine_core():
        """Фрагмент –N< : третичный аминный центр.
        Три связи к C устанавливаются в алгоритме сборки.
        """
        return np.array([[0]], dtype=int)

    @staticmethod
    def get_amide_core():
        """Амидное ядро –CONH– (C–O–N).

        Атомы: C0, O1, N2. Связи: C0=O1, C0–N2.
        Внешние связи: C0 к углеродному скелету, N2 к одному C (или двум для N-замещённых амидов).
        """
        return np.array([
            [0, 2, 1],
            [2, 0, 0],
            [1, 0, 0]
        ], dtype=int)

    @staticmethod
    def get_nitrile():
        """Фрагмент –C#N (нитрил).

        Атомы: C0, N1, связь тройная. C0 связывается с углеродным скелетом.
        """
        return np.array([
            [0, 3],
            [3, 0]
        ], dtype=int)

    # 4) Простые гетероциклы (только тяжёлые атомы, H неявные)

    @staticmethod
    def get_pyridine():
        """Пиридин: 6-членное ароматическое кольцо C5N1.

        Нумерация: N0, C1–C5 по кольцу.
        Связи: как бензол, но один C заменён на N.
        """
        return np.array([
            [0, 2, 0, 0, 0, 1],  # N0
            [2, 0, 1, 0, 0, 0],
            [0, 1, 0, 2, 0, 0],
            [0, 0, 2, 0, 1, 0],
            [0, 0, 0, 1, 0, 2],
            [1, 0, 0, 0, 2, 0]
        ], dtype=int)

    @staticmethod
    def get_pyrimidine():
        """Пиримидин: 6-членное ароматическое кольцо C4N2.

        Нумерация: N0, C1, N2, C3, C4, C5 по кольцу.
        Матрица по типу пиридина, но с двумя N.
        """
        return np.array([
            [0, 2, 0, 0, 0, 1],  # N0
            [2, 0, 1, 0, 0, 0],  # C1
            [0, 1, 0, 2, 0, 0],  # N2
            [0, 0, 2, 0, 1, 0],  # C3
            [0, 0, 0, 1, 0, 2],  # C4
            [1, 0, 0, 0, 2, 0]   # C5
        ], dtype=int)

    @staticmethod
    def get_pyrrole():
        """Пиррол: 5-членный ароматический цикл C4N1 (N с одним H, H неявный).

        Нумерация: N0, C1, C2, C3, C4 по кольцу.
        Связи: чередование 2/1 как в ароматическом 5-членном кольце.
        """
        return np.array([
            [0, 1, 0, 0, 2],  # N0
            [1, 0, 2, 0, 0],
            [0, 2, 0, 1, 0],
            [0, 0, 1, 0, 2],
            [2, 0, 0, 2, 0]
        ], dtype=int)

    @staticmethod
    def get_morpholine():
        """Морфолин: 6-членный насыщенный цикл C4N1O1.

        Нумерация: N0, C1, C2, O3, C4, C5 по кольцу. Все связи одинарные.
        """
        return np.array([
            [0, 1, 0, 0, 0, 1],  # N0-C1, N0-C5
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0]
        ], dtype=int)

    # 5) Ароматические кислородсодержащие фрагменты

    @staticmethod
    def get_phenol_core():
        """Фенольное ядро: бензол, где один C помечен под –OH.

        Здесь возвращается просто матрица бензола; информация о том,
        какой атом замещён OH, должна храниться в логике генератора
        (например, через attachment_points).
        """
        return FragmentLibrary.get_benzene().copy()

    @staticmethod
    def get_aryl_carbonyl_core():
        """Ароматический карбонил –CO–, связывающийся с ароматическим C.

        Фактически это тот же карбонильный фрагмент C=O,
        но его можно учитывать отдельно в словаре фрагментов.
        """
        return FragmentLibrary.get_carbonyl().copy()



class MoleculeFragment:
    """Фрагмент молекулы с пронумерованными вершинами и внутренней структурой."""

    def __init__(self, name: str, heavy_formula: Dict[str, int], ihd: float,
                 atoms: List[str], bonds: List[Tuple[int, int, int]],
                 attachment_points: List[int]):
        self.name = name
        self.heavy_formula = heavy_formula
        self.ihd = ihd
        self.atoms = atoms
        self.bonds = bonds
        self.attachment_points = attachment_points.copy()
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> Dict[int, List[Tuple[int, int]]]:
        adj = defaultdict(list)
        for i, j, order in self.bonds:
            adj[i].append((j, order))
            adj[j].append((i, order))
        return dict(adj)

    def get_num_atoms(self) -> int:
        return len(self.atoms)

    def get_free_attachment_points(self) -> List[int]:
        return self.attachment_points.copy()

    def has_free_attachment_point(self, idx: int) -> bool:
        return idx in self.attachment_points

    def connect_to(self, other: 'MoleculeFragment',
                   my_point: int, other_point: int,
                   bond_order: int = 1) -> 'MoleculeFragment':
        if not self.has_free_attachment_point(my_point):
            raise ValueError(f"Точка {my_point} в {self.name} уже занята")
        if not other.has_free_attachment_point(other_point):
            raise ValueError(f"Точка {other_point} в {other.name} уже занята")

        new_name = f"{self.name}+{other.name}"
        new_heavy = self.heavy_formula.copy()
        for el, count in other.heavy_formula.items():
            new_heavy[el] = new_heavy.get(el, 0) + count

        new_ihd = self.ihd + other.ihd
        offset = len(self.atoms)
        new_atoms = self.atoms.copy()
        new_atoms.extend(other.atoms)

        new_bonds = self.bonds.copy()
        for i, j, order in other.bonds:
            new_bonds.append((i + offset, j + offset, order))
        new_bonds.append((my_point, other_point + offset, bond_order))

        new_attachment_points = []
        for pt in self.attachment_points:
            if pt != my_point:
                new_attachment_points.append(pt)
        for pt in other.attachment_points:
            if pt != other_point:
                new_attachment_points.append(pt + offset)

        return MoleculeFragment(
            name=new_name,
            heavy_formula=new_heavy,
            ihd=new_ihd,
            atoms=new_atoms,
            bonds=new_bonds,
            attachment_points=new_attachment_points
        )

    def __repr__(self) -> str:
        return (f"MoleculeFragment(name='{self.name}', "
                f"formula={self.heavy_formula}, ihd={self.ihd}, "
                f"atoms={len(self.atoms)}, bonds={len(self.bonds)}, "
                f"free_points={len(self.attachment_points)})")


from typing import Dict, List, Tuple
from collections import defaultdict

class MoleculeFragment:
    def __init__(self, name: str, heavy_formula: Dict[str, int], ihd: float,
                 atoms: List[str], bonds: List[Tuple[int, int, int]],
                 attachment_points: List[int]):
        self.name = name
        self.heavy_formula = heavy_formula
        self.ihd = ihd
        self.atoms = atoms
        self.bonds = bonds
        self.attachment_points = attachment_points.copy()
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self):
        adj = defaultdict(list)
        for i, j, order in self.bonds:
            adj[i].append((j, order))
            adj[j].append((i, order))
        return dict(adj)

    def get_num_atoms(self): return len(self.atoms)
    def get_free_attachment_points(self): return self.attachment_points.copy()
    def has_free_attachment_point(self, idx): return idx in self.attachment_points

    def connect_to(self, other, my_point, other_point, bond_order=1):
        if not self.has_free_attachment_point(my_point):
            raise ValueError(f"Точка {my_point} в {self.name} занята")
        if not other.has_free_attachment_point(other_point):
            raise ValueError(f"Точка {other_point} в {other.name} занята")

        new_heavy = self.heavy_formula.copy()
        for el, count in other.heavy_formula.items():
            new_heavy[el] = new_heavy.get(el, 0) + count

        offset = len(self.atoms)
        new_atoms = self.atoms + other.atoms
        new_bonds = self.bonds.copy()
        for i, j, order in other.bonds:
            new_bonds.append((i + offset, j + offset, order))
        new_bonds.append((my_point, other_point + offset, bond_order))

        new_points = [p for p in self.attachment_points if p != my_point]
        new_points += [p + offset for p in other.attachment_points if p != other_point]

        return MoleculeFragment(f"{self.name}+{other.name}", new_heavy,
                               self.ihd + other.ihd, new_atoms, new_bonds, new_points)

    def __repr__(self):
        return f"MoleculeFragment('{self.name}', {self.heavy_formula}, IHD={self.ihd})"


# === АЦИКЛИЧЕСКИЕ ФРАГМЕНТЫ ===
def create_methylene(): return MoleculeFragment('methylene', {'C': 1}, 0, ['C'], [], [0, 0])
def create_ethylene(): return MoleculeFragment('ethylene', {'C': 2}, 0, ['C', 'C'], [(0, 1, 1)], [0, 1])
def create_propylene(): return MoleculeFragment('propylene', {'C': 3}, 0, ['C']*3, [(0,1,1),(1,2,1)], [0, 2])
def create_alkene(): return MoleculeFragment('alkene', {'C': 2}, 1, ['C', 'C'], [(0, 1, 2)], [0, 1])
def create_propenyl(): return MoleculeFragment('propenyl', {'C': 3}, 1, ['C']*3, [(0,1,2),(1,2,1)], [0, 2])
def create_alkyne(): return MoleculeFragment('alkyne', {'C': 2}, 2, ['C', 'C'], [(0, 1, 3)], [0, 1])
def create_propynyl(): return MoleculeFragment('propynyl', {'C': 3}, 2, ['C']*3, [(0,1,3),(1,2,1)], [0, 2])

# === 5-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
def create_cyclopentane(): return MoleculeFragment('cyclopentane', {'C': 5}, 1, ['C']*5, [(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,0,1)], list(range(5)))
def create_cyclopentene(): return MoleculeFragment('cyclopentene', {'C': 5}, 2, ['C']*5, [(0,1,2),(1,2,1),(2,3,1),(3,4,1),(4,0,1)], list(range(5)))
def create_cyclopentadiene(): return MoleculeFragment('cyclopentadiene', {'C': 5}, 3, ['C']*5, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,0,1)], list(range(5)))

# === 6-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
def create_cyclohexane(): return MoleculeFragment('cyclohexane', {'C': 6}, 1, ['C']*6, [(i,(i+1)%6,1) for i in range(6)], list(range(6)))
def create_cyclohexene(): return MoleculeFragment('cyclohexene', {'C': 6}, 2, ['C']*6, [(0,1,2)]+[(i,(i+1)%6,1) for i in range(1,6)], list(range(6)))
def create_cyclohexadiene(): return MoleculeFragment('cyclohexadiene', {'C': 6}, 3, ['C']*6, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,1),(5,0,1)], list(range(6)))
def create_benzene(): return MoleculeFragment('benzene', {'C': 6}, 4, ['C']*6, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,0,1)], list(range(6)))

# === 8-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
def create_cyclooctane(): return MoleculeFragment('cyclooctane', {'C': 8}, 1, ['C']*8, [(i,(i+1)%8,1) for i in range(8)], list(range(8)))
def create_cyclooctene(): return MoleculeFragment('cyclooctene', {'C': 8}, 2, ['C']*8, [(0,1,2)]+[(i,(i+1)%8,1) for i in range(1,8)], list(range(8)))
def create_cyclooctadiene(): return MoleculeFragment('cyclooctadiene', {'C': 8}, 3, ['C']*8, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,1),(5,6,1),(6,7,1),(7,0,1)], list(range(8)))
def create_cyclooctatriene(): return MoleculeFragment('cyclooctatriene', {'C': 8}, 4, ['C']*8, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,6,1),(6,7,1),(7,0,1)], list(range(8)))
def create_cyclooctatetraene(): return MoleculeFragment('cyclooctatetraene', {'C': 8}, 5, ['C']*8, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,6,1),(6,7,2),(7,0,1)], list(range(8)))

# === 10-ЧЛЕННЫЕ УГЛЕРОДНЫЕ ЦИКЛЫ ===
def create_cyclodecane(): return MoleculeFragment('cyclodecane', {'C': 10}, 1, ['C']*10, [(i,(i+1)%10,1) for i in range(10)], list(range(10)))
def create_cyclodecene(): return MoleculeFragment('cyclodecene', {'C': 10}, 2, ['C']*10, [(0,1,2)]+[(i,(i+1)%10,1) for i in range(1,10)], list(range(10)))
def create_cyclodecadiene(): return MoleculeFragment('cyclodecadiene', {'C': 10}, 3, ['C']*10, [(0,1,2),(2,3,2)]+[(i,(i+1)%10,1) for i in [1]+list(range(3,10))], list(range(10)))
def create_cyclodecatriene(): return MoleculeFragment('cyclodecatriene', {'C': 10}, 4, ['C']*10, [(0,1,2),(2,3,2),(4,5,2)]+[(i,(i+1)%10,1) for i in [1,3]+list(range(5,10))], list(range(10)))
def create_cyclodecatetraene(): return MoleculeFragment('cyclodecatetraene', {'C': 10}, 5, ['C']*10, [(0,1,2),(2,3,2),(4,5,2),(6,7,2)]+[(i,(i+1)%10,1) for i in [1,3,5]+list(range(7,10))], list(range(10)))
def create_cyclodecapentaene(): return MoleculeFragment('cyclodecapentaene', {'C': 10}, 6, ['C']*10, [(i,(i+1)%10,2 if i%2==0 else 1) for i in range(10)], list(range(10)))

# === КОНДЕНСИРОВАННЫЕ СИСТЕМЫ ===
def create_naphthalene(): return MoleculeFragment('naphthalene', {'C': 10}, 7, ['C']*10, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,6,1),(6,7,2),(7,8,1),(8,9,2),(9,0,1),(4,9,1)], [0,1,2,3,5,6,7,8])
def create_anthracene(): return MoleculeFragment('anthracene', {'C': 14}, 10, ['C']*14, [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,6,1),(6,7,2),(7,8,1),(8,9,2),(9,10,1),(10,11,2),(11,12,1),(12,13,2),(13,0,1),(4,13,1),(8,12,1)], [0,1,2,3,5,6,7,9,10,11])

# === 5-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С АЗОТОМ ===
def create_pyrrolidine(): return MoleculeFragment('pyrrolidine', {'C': 4, 'N': 1}, 1, ['C','C','C','C','N'], [(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,0,1)], list(range(5)))
def create_pyrroline(): return MoleculeFragment('pyrroline', {'C': 4, 'N': 1}, 2, ['C','C','C','C','N'], [(0,1,2),(1,2,1),(2,3,1),(3,4,1),(4,0,1)], list(range(5)))
def create_pyrrole(): return MoleculeFragment('pyrrole', {'C': 4, 'N': 1}, 3, ['C','C','C','C','N'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,0,1)], list(range(5)))
def create_imidazole(): return MoleculeFragment('imidazole', {'C': 3, 'N': 2}, 3, ['C','N','C','N','C'], [(0,1,1),(1,2,2),(2,3,1),(3,4,2),(4,0,1)], list(range(5)))
def create_pyrazole(): return MoleculeFragment('pyrazole', {'C': 3, 'N': 2}, 3, ['C','N','N','C','C'], [(0,1,1),(1,2,1),(2,3,2),(3,4,1),(4,0,2)], list(range(5)))

# === 5-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С КИСЛОРОДОМ ===
def create_tetrahydrofuran(): return MoleculeFragment('tetrahydrofuran', {'C': 4, 'O': 1}, 1, ['C','C','C','C','O'], [(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,0,1)], [0,1,2,3])
def create_dihydrofuran(): return MoleculeFragment('dihydrofuran', {'C': 4, 'O': 1}, 2, ['C','C','C','C','O'], [(0,1,2),(1,2,1),(2,3,1),(3,4,1),(4,0,1)], [0,1,2,3])
def create_furan(): return MoleculeFragment('furan', {'C': 4, 'O': 1}, 3, ['C','C','C','C','O'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,0,1)], [0,1,2,3])

# === 6-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С АЗОТОМ ===
def create_piperidine(): return MoleculeFragment('piperidine', {'C': 5, 'N': 1}, 1, ['C']*5+['N'], [(i,(i+1)%6,1) for i in range(6)], list(range(6)))
def create_tetrahydropyridine(): return MoleculeFragment('tetrahydropyridine', {'C': 5, 'N': 1}, 2, ['C']*5+['N'], [(0,1,2)]+[(i,(i+1)%6,1) for i in range(1,6)], list(range(6)))
def create_dihydropyridine(): return MoleculeFragment('dihydropyridine', {'C': 5, 'N': 1}, 3, ['C']*5+['N'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,1),(5,0,1)], list(range(6)))
def create_pyridine(): return MoleculeFragment('pyridine', {'C': 5, 'N': 1}, 4, ['C']*5+['N'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,0,1)], list(range(6)))
def create_pyrimidine(): return MoleculeFragment('pyrimidine', {'C': 4, 'N': 2}, 4, ['C','N','C','N','C','C'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,0,1)], list(range(6)))
def create_pyrazine(): return MoleculeFragment('pyrazine', {'C': 4, 'N': 2}, 4, ['C','N','C','C','N','C'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,2),(5,0,1)], list(range(6)))
def create_pyridazine(): return MoleculeFragment('pyridazine', {'C': 4, 'N': 2}, 4, ['C']*4+['N','N'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,1),(5,0,2)], list(range(6)))

# === 6-ЧЛЕННЫЕ ГЕТЕРОЦИКЛЫ С КИСЛОРОДОМ ===
def create_tetrahydropyran(): return MoleculeFragment('tetrahydropyran', {'C': 5, 'O': 1}, 1, ['C']*5+['O'], [(i,(i+1)%6,1) for i in range(6)], [0,1,2,3,4])
def create_dihydropyran(): return MoleculeFragment('dihydropyran', {'C': 5, 'O': 1}, 2, ['C']*5+['O'], [(0,1,2)]+[(i,(i+1)%6,1) for i in range(1,6)], [0,1,2,3,4])
def create_pyran(): return MoleculeFragment('pyran', {'C': 5, 'O': 1}, 3, ['C']*5+['O'], [(0,1,2),(1,2,1),(2,3,2),(3,4,1),(4,5,1),(5,0,1)], [0,1,2,3,4])

# === ФУНКЦИОНАЛЬНЫЕ ГРУППЫ ===
def create_cooh(): return MoleculeFragment('cooh', {'C': 1, 'O': 2}, 1, ['C','O','O'], [(0,1,2),(0,2,1)], [0])
def create_oh(): return MoleculeFragment('oh', {'O': 1}, 0, ['O'], [], [0])
def create_cho(): return MoleculeFragment('cho', {'C': 1, 'O': 1}, 1, ['C','O'], [(0,1,2)], [0])
def create_co(): return MoleculeFragment('co', {'C': 1, 'O': 1}, 1, ['C','O'], [(0,1,2)], [0,0])
def create_coo(): return MoleculeFragment('coo', {'C': 1, 'O': 2}, 1, ['C','O','O'], [(0,1,2),(0,2,1)], [0,2])
def create_o_ether(): return MoleculeFragment('o_ether', {'O': 1}, 0, ['O'], [], [0,0])
def create_nh2(): return MoleculeFragment('nh2', {'N': 1}, 0, ['N'], [], [0])
def create_nh(): return MoleculeFragment('nh', {'N': 1}, 0, ['N'], [], [0,0])
def create_n_tertiary(): return MoleculeFragment('n_tertiary', {'N': 1}, 0, ['N'], [], [0,0,0])
def create_no2(): return MoleculeFragment('no2', {'N': 1, 'O': 2}, 1, ['N','O','O'], [(0,1,2),(0,2,2)], [0])
def create_cn(): return MoleculeFragment('cn', {'C': 1, 'N': 1}, 2, ['C','N'], [(0,1,3)], [0])
def create_conh2(): return MoleculeFragment('conh2', {'C': 1, 'O': 1, 'N': 1}, 1, ['C','O','N'], [(0,1,2),(0,2,1)], [0])
def create_sh(): return MoleculeFragment('sh', {'S': 1}, 0, ['S'], [], [0])
def create_s_sulfide(): return MoleculeFragment('s_sulfide', {'S': 1}, 0, ['S'], [], [0,0])
def create_so2(): return MoleculeFragment('so2', {'S': 1, 'O': 2}, 0, ['S','O','O'], [(0,1,2),(0,2,2)], [0,0])
def create_so3h(): return MoleculeFragment('so3h', {'S': 1, 'O': 3}, 0, ['S','O','O','O'], [(0,1,2),(0,2,2),(0,3,1)], [0])
def create_f(): return MoleculeFragment('f', {'F': 1}, 0, ['F'], [], [0])
def create_cl(): return MoleculeFragment('cl', {'Cl': 1}, 0, ['Cl'], [], [0])
def create_br(): return MoleculeFragment('br', {'Br': 1}, 0, ['Br'], [], [0])
def create_i(): return MoleculeFragment('i', {'I': 1}, 0, ['I'], [], [0])

# Словарь всех фабричных функций
ALL_FRAGMENTS = {
    'methylene': create_methylene, 'ethylene': create_ethylene, 'propylene': create_propylene,
    'alkene': create_alkene, 'propenyl': create_propenyl, 'alkyne': create_alkyne, 'propynyl': create_propynyl,
    'cyclopentane': create_cyclopentane, 'cyclopentene': create_cyclopentene, 'cyclopentadiene': create_cyclopentadiene,
    'cyclohexane': create_cyclohexane, 'cyclohexene': create_cyclohexene, 'cyclohexadiene': create_cyclohexadiene, 'benzene': create_benzene,
    'cyclooctane': create_cyclooctane, 'cyclooctene': create_cyclooctene, 'cyclooctadiene': create_cyclooctadiene,
    'cyclooctatriene': create_cyclooctatriene, 'cyclooctatetraene': create_cyclooctatetraene,
    'cyclodecane': create_cyclodecane, 'cyclodecene': create_cyclodecene, 'cyclodecadiene': create_cyclodecadiene,
    'cyclodecatriene': create_cyclodecatriene, 'cyclodecatetraene': create_cyclodecatetraene, 'cyclodecapentaene': create_cyclodecapentaene,
    'naphthalene': create_naphthalene, 'anthracene': create_anthracene,
    'pyrrolidine': create_pyrrolidine, 'pyrroline': create_pyrroline, 'pyrrole': create_pyrrole,
    'imidazole': create_imidazole, 'pyrazole': create_pyrazole,
    'tetrahydrofuran': create_tetrahydrofuran, 'dihydrofuran': create_dihydrofuran, 'furan': create_furan,
    'piperidine': create_piperidine, 'tetrahydropyridine': create_tetrahydropyridine, 'dihydropyridine': create_dihydropyridine,
    'pyridine': create_pyridine, 'pyrimidine': create_pyrimidine, 'pyrazine': create_pyrazine, 'pyridazine': create_pyridazine,
    'tetrahydropyran': create_tetrahydropyran, 'dihydropyran': create_dihydropyran, 'pyran': create_pyran,
    'cooh': create_cooh, 'oh': create_oh, 'cho': create_cho, 'co': create_co, 'coo': create_coo,
    'o_ether': create_o_ether, 'nh2': create_nh2, 'nh': create_nh, 'n_tertiary': create_n_tertiary,
    'no2': create_no2, 'cn': create_cn, 'conh2': create_conh2,
    'sh': create_sh, 's_sulfide': create_s_sulfide, 'so2': create_so2, 'so3h': create_so3h,
    'f': create_f, 'cl': create_cl, 'br': create_br, 'i': create_i,
}


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

def filter_fragments(target_heavy, target_ihd, fragment_library):
    filtered = {}
    for name, f in fragment_library.items():
        hf = f["heavy_formula"]
        ihd = f["ihd"]
        # отсев по IHD
        if ihd > target_ihd:
            continue
        # отсев по элементам
        bad = False
        for el, n in hf.items():
            if el not in target_heavy or n > target_heavy[el]:
                bad = True
                break
        if bad:
            continue
        filtered[name] = f
    return filtered

def find_fragment_combinations(target_heavy_formula, target_ihd,
                               num_cooh=0, num_oh=0,
                               max_bases=10):
    results = []

    # учёт функциональных групп
    func_heavy = {}
    func_ihd = 0
    if num_cooh:
        add_formula(func_heavy, FUNCTIONAL_GROUPS["cooh"]["heavy_formula"], num_cooh)
        func_ihd += FUNCTIONAL_GROUPS["cooh"]["ihd"] * num_cooh
    if num_oh:
        add_formula(func_heavy, FUNCTIONAL_GROUPS["oh"]["heavy_formula"], num_oh)
        func_ihd += FUNCTIONAL_GROUPS["oh"]["ihd"] * num_oh

    # скорректированная цель: что должны дать только базовые фрагменты
    base_target = target_heavy_formula.copy()
    for el, n in func_heavy.items():
        base_target[el] = base_target.get(el, 0) - n
        if base_target[el] < 0:
            return []  # функционалки уже «перебили» формулу

    base_target = {el: n for el, n in base_target.items() if n > 0}

    base_target_ihd = target_ihd - func_ihd
    if base_target_ihd < 0:
        return []

    # усечённая библиотека
    lib = filter_fragments(base_target, base_target_ihd, FRAGMENT_LIBRARY)
    names = sorted(lib.keys())

    def backtrack(idx, current_counts, current_heavy, current_ihd, used_bases):
        # отсев по числу баз
        if used_bases > max_bases:
            return
        # отсев по формуле / IHD (верхняя граница)
        for el, n in current_heavy.items():
            if n > base_target.get(el, 0):
                return
        if current_ihd > base_target_ihd + 1e-6:
            return

        # если прошли все фрагменты — проверяем точное совпадение
        if idx == len(names):
            if current_heavy == base_target and abs(current_ihd - base_target_ihd) < 1e-6:
                bases_dict = {names[i]: c for i, c in enumerate(current_counts) if c > 0}
                results.append({
                    "bases": bases_dict,
                    "cooh": num_cooh,
                    "oh": num_oh,
                    "total_heavy_formula": target_heavy_formula.copy(),
                    "total_ihd": target_ihd,
                })
            return

        name = names[idx]
        frag = lib[name]
        hf = frag["heavy_formula"]
        ihd_f = frag["ihd"]

        # оценка максимального допустимого количества этого фрагмента по каждому элементу и IHD
        max_by_elem = float("inf")
        for el, n in hf.items():
            if n > 0:
                rem = base_target.get(el, 0) - current_heavy.get(el, 0)
                max_by_elem = min(max_by_elem, rem // n)
        if ihd_f > 0:
            max_by_ihd = int((base_target_ihd - current_ihd) // ihd_f)
            max_mult = min(max_by_elem, max_by_ihd)
        else:
            max_mult = max_by_elem

        if max_mult == float("inf"):
            max_mult = 0

        # перебираем 0..max_mult копий текущего фрагмента
        for k in range(max_mult + 1):
            # добавляем k копий
            new_heavy = current_heavy
            new_ihd = current_ihd
            if k > 0:
                new_heavy = current_heavy.copy()
                for el, n in hf.items():
                    new_heavy[el] = new_heavy.get(el, 0) + n * k
                new_ihd = current_ihd + ihd_f * k

            current_counts[idx] = k
            backtrack(idx + 1, current_counts, new_heavy, new_ihd, used_bases + k)

        current_counts[idx] = 0  # на всякий случай

    current_counts = [0] * len(names)
    backtrack(0, current_counts, {}, 0.0, 0)

    return results
def assemble_molecule_from_combination(combination: dict,
                                       fragment_library_dict: dict = None) -> MoleculeFragment:
    """Собирает полную молекулу из комбинации фрагментов.

    Процесс сборки:
    1. Выделяет базовые фрагменты (всё кроме COOH и OH)
    2. Последовательно соединяет базовые фрагменты
    3. Добавляет COOH группы на свободные точки
    4. Добавляет OH группы на свободные точки

    Args:
        combination: словарь с результатом find_fragment_combinations
                    {'bases': {'benzene': 1}, 'cooh': 1, 'oh': 0, ...}
        fragment_library_dict: словарь {name: factory_function}
                              По умолчанию использует ALL_FRAGMENTS

    Returns:
        MoleculeFragment - собранная молекула

    Raises:
        ValueError: если не хватает свободных точек присоединения
    """
    if fragment_library_dict is None:
        fragment_library_dict = ALL_FRAGMENTS

    # Извлекаем информацию из комбинации
    bases = combination.get('bases', {})
    num_cooh = combination.get('cooh', 0)
    num_oh = combination.get('oh', 0)

    # === ШАГ 1: Создаём базовые фрагменты ===
    base_fragments = []
    for name, count in bases.items():
        if name not in fragment_library_dict:
            raise ValueError(f"Фрагмент '{name}' не найден в библиотеке")
        for _ in range(count):
            base_fragments.append(fragment_library_dict[name]())

    if not base_fragments and not num_cooh and not num_oh:
        raise ValueError("Комбинация не содержит фрагментов")

    # === ШАГ 2: Последовательно соединяем базовые фрагменты ===
    if base_fragments:
        current = base_fragments[0]

        for next_frag in base_fragments[1:]:
            # Находим свободные точки
            my_points = current.get_free_attachment_points()
            other_points = next_frag.get_free_attachment_points()

            if not my_points or not other_points:
                raise ValueError(f"Нет свободных точек для соединения {current.name} и {next_frag.name}")

            # Соединяем через первые доступные точки
            current = current.connect_to(next_frag, my_points[0], other_points[0], bond_order=1)
    else:
        # Если нет базовых фрагментов, начинаем с первой COOH группы
        current = create_cooh()
        num_cooh -= 1

    # === ШАГ 3: Добавляем COOH группы ===
    for i in range(num_cooh):
        free_points = current.get_free_attachment_points()
        if not free_points:
            raise ValueError(f"Не хватает свободных точек для добавления COOH группы #{i+1}")

        cooh = create_cooh()
        current = current.connect_to(cooh, free_points[0], 0, bond_order=1)

    # === ШАГ 4: Добавляем OH группы ===
    for i in range(num_oh):
        free_points = current.get_free_attachment_points()
        if not free_points:
            raise ValueError(f"Не хватает свободных точек для добавления OH группы #{i+1}")

        oh = create_oh()
        current = current.connect_to(oh, free_points[0], 0, bond_order=1)

    return current


def assemble_all_combinations(combinations: list,
                              fragment_library_dict: dict = None) -> list:
    """Собирает молекулы из всех найденных комбинаций.

    Args:
        combinations: список результатов find_fragment_combinations
        fragment_library_dict: словарь фабричных функций

    Returns:
        Список собранных MoleculeFragment объектов
    """
    if fragment_library_dict is None:
        fragment_library_dict = ALL_FRAGMENTS

    molecules = []
    for i, combo in enumerate(combinations):
        try:
            mol = assemble_molecule_from_combination(combo, fragment_library_dict)
            molecules.append({
                'index': i,
                'combination': combo,
                'molecule': mol,
                'success': True
            })
        except Exception as e:
            molecules.append({
                'index': i,
                'combination': combo,
                'molecule': None,
                'success': False,
                'error': str(e)
            })

    return molecules



def find_and_visualize_molecules(brutto_formula: str,
                                 num_cooh: int = 0,
                                 num_oh: int = 0,
                                 max_bases: int = 10,
                                 show_images: bool = True,
                                 image_size: tuple = (400, 300)):
    """Итоговая функция: от брутто-формулы до визуализации молекул.

    Выполняет полный цикл:
    1. Парсит брутто-формулу
    2. Вычисляет IHD
    3. Находит все возможные комбинации фрагментов
    4. Собирает молекулы из комбинаций
    5. Визуализирует структуры (если установлен RDKit)

    Args:
        brutto_formula: брутто-формула (например, "C7H6O2")
        num_cooh: количество COOH групп
        num_oh: количество OH групп
        max_bases: максимальное количество базовых фрагментов
        show_images: показывать ли изображения (требуется RDKit)
        image_size: размер изображений (ширина, высота)

    Returns:
        dict с ключами:
            - 'input': входные данные
            - 'heavy_formula': формула тяжёлых атомов
            - 'ihd': индекс ненасыщенности
            - 'combinations': найденные комбинации фрагментов
            - 'molecules': список собранных молекул с метаданными
            - 'images': список PIL изображений (если show_images=True)

    Пример:
        result = find_and_visualize_molecules("C7H6O2", num_cooh=1, num_oh=0)
        print(f"Найдено {len(result['molecules'])} структур")
        for mol in result['molecules']:
        print(f"  - {mol['name']}: {mol['formula']}")
    """

    # === ШАГ 2: Вычисление тяжёлой формулы и IHD ===
    full_formula = parse_formula(brutto_formula)

    # Убираем водороды для тяжёлой формулы
    heavy_formula = {k: v for k, v in full_formula.items() if k != 'H'}

    # Вычисляем IHD по формуле: IHD = (2C + 2 - H + N) / 2
    C = full_formula.get('C', 0)
    H = full_formula.get('H', 0)
    N = full_formula.get('N', 0)

    ihd = (2 * C + 2 - H + N) / 2

    print(f"📋 Исходные данные:")
    print(f"   Брутто-формула: {brutto_formula}")
    print(f"   Тяжёлая формула: {heavy_formula}")
    print(f"   IHD: {ihd}")
    print(f"   COOH групп: {num_cooh}")
    print(f"   OH групп: {num_oh}")
    print()

    # === ШАГ 3: Поиск комбинаций ===
    print("🔍 Поиск возможных комбинаций фрагментов...")

    combinations = find_fragment_combinations(
    target_heavy_formula=heavy_formula,
    target_ihd=ihd,
    num_cooh=num_cooh,
    num_oh=num_oh,
    max_bases=max_bases
        )
    print(f"✅ Найдено {len(combinations)} комбинаций")


    if not combinations:
        print("⚠️  Подходящих комбинаций не найдено")
        return {
            'input': {'brutto': brutto_formula, 'cooh': num_cooh, 'oh': num_oh},
            'heavy_formula': heavy_formula,
            'ihd': ihd,
            'combinations': [],
            'molecules': [],
            'images': []
        }

    # === ШАГ 4: Сборка молекул ===
    print("\n🔧 Сборка молекул из комбинаций...")
    assembled = assemble_all_combinations(combinations)

    successful = [r for r in assembled if r['success']]
    failed = [r for r in assembled if not r['success']]

    print(f"✅ Успешно собрано: {len(successful)}")
    if failed:
        print(f"❌ Не удалось собрать: {len(failed)}")

    # === ШАГ 5: Подготовка результата ===
    molecules_data = []
    images = []

    for result in successful:
        mol = result['molecule']
        combo = result['combination']

        mol_info = {
            'index': result['index'],
            'name': mol.name,
            'formula': mol.heavy_formula,
            'ihd': mol.ihd,
            'num_atoms': mol.get_num_atoms(),
            'num_bonds': len(mol.bonds),
            'free_points': len(mol.get_free_attachment_points()),
            'combination': combo,
            'fragment_object': mol
        }
        molecules_data.append(mol_info)

    # === ШАГ 6: Визуализация (если требуется) ===
    if show_images:
        print("\n🎨 Визуализация структур...")
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw, AllChem

            for mol_data in molecules_data:
                mol_obj = mol_data['fragment_object']

                # Создаём RDKit молекулу
                rdkit_mol = Chem.RWMol()
                for symbol in mol_obj.atoms:
                    rdkit_mol.AddAtom(Chem.Atom(symbol))

                for i, j, order in mol_obj.bonds:
                    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
                    rdkit_mol.AddBond(i, j, bond_types[order-1])

                rdkit_mol = rdkit_mol.GetMol()

                # Санитизация
                try:
                    Chem.SanitizeMol(rdkit_mol)
                except:
                    try:
                        Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
                    except:
                        pass

                # Добавляем водороды и генерируем координаты
                rdkit_mol = Chem.AddHs(rdkit_mol)
                AllChem.Compute2DCoords(rdkit_mol)

                # Генерируем изображение
                img = Draw.MolToImage(rdkit_mol, size=image_size)
                images.append(img)
                mol_data['image'] = img

            print(f"✅ Создано {len(images)} изображений")

        except ImportError:
            print("⚠️  RDKit не установлен, визуализация недоступна")
            print("   Установите: pip install rdkit")

    # === ШАГ 7: Вывод результатов ===
    print("\n" + "="*60)
    print(f"📊 ИТОГО: найдено {len(molecules_data)} структур для {brutto_formula}")
    print("="*60)

    for i, mol_data in enumerate(molecules_data, 1):
        print(f"\n{i}. {mol_data['name']}")
        print(f"   Формула: {mol_data['formula']}")
        print(f"   IHD: {mol_data['ihd']}")
        print(f"   Фрагменты: {mol_data['combination']['bases']}")
        print(f"   COOH: {mol_data['combination']['cooh']}, OH: {mol_data['combination']['oh']}")

    print("\n" + "="*60)

    return {
        'input': {
            'brutto': brutto_formula,
            'cooh': num_cooh,
            'oh': num_oh,
            'max_bases': max_bases
        },
        'heavy_formula': heavy_formula,
        'ihd': ihd,
        'combinations': combinations,
        'molecules': molecules_data,
        'images': images
    }


def to_rdkit_mol(fragment: MoleculeFragment):
    """Конвертирует MoleculeFragment в RDKit Mol объект."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.RWMol()

    # Добавляем атомы
    atom_indices = {}
    for i, symbol in enumerate(fragment.atoms):
        atom = Chem.Atom(symbol)
        idx = mol.AddAtom(atom)
        atom_indices[i] = idx

    # Добавляем связи
    for i, j, order in fragment.bonds:
        if order == 1:
            bond_type = Chem.BondType.SINGLE
        elif order == 2:
            bond_type = Chem.BondType.DOUBLE
        elif order == 3:
            bond_type = Chem.BondType.TRIPLE
        else:
            bond_type = Chem.BondType.SINGLE

        mol.AddBond(atom_indices[i], atom_indices[j], bond_type)

    # Конвертируем в Mol
    mol = mol.GetMol()

    # ИСПРАВЛЕНИЕ: Санитизируем молекулу перед добавлением водородов
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    except:
        # Если полная санитизация не удалась, пробуем базовую
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS |
                                          Chem.SANITIZE_SETAROMATICITY |
                                          Chem.SANITIZE_SETCONJUGATION)

    # Теперь можем добавить водороды
    mol = Chem.AddHs(mol)

    # Генерируем 2D координаты
    AllChem.Compute2DCoords(mol)

    return mol


def visualize_fragment(fragment: MoleculeFragment,
                       highlight_attachment_points: bool = True,
                       size: tuple = (400, 300)):
    """Визуализирует фрагмент с помощью RDKit."""
    from rdkit.Chem import Draw

    mol = to_rdkit_mol(fragment)

    highlight_atoms = []
    if highlight_attachment_points and fragment.attachment_points:
        highlight_atoms = fragment.attachment_points

    img = Draw.MolToImage(
        mol,
        size=size,
        highlightAtoms=highlight_atoms,
        highlightColor=(0.8, 1.0, 0.8)
    )

    return img


def visualize_fragments_grid(fragments: list,
                             names: list = None,
                             mols_per_row: int = 3,
                             subImgSize: tuple = (300, 250)):
    """Визуализирует несколько фрагментов в виде сетки."""
    from rdkit.Chem import Draw

    mols = [to_rdkit_mol(frag) for frag in fragments]

    if names is None:
        names = [f.name for f in fragments]

    legends = []
    for i, frag in enumerate(fragments):
        formula_str = ''.join(f"{el}{n if n > 1 else ''}"
                             for el, n in sorted(frag.heavy_formula.items()))
        legend = f"{names[i]}\n{formula_str}, IHD={frag.ihd}"
        legends.append(legend)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=subImgSize,
        legends=legends
    )

    return img


def visualize_connection_sequence(fragments: list,
                                  connections: list,
                                  size: tuple = (400, 300)):
    """Визуализирует последовательность соединения фрагментов."""
    images = []

    current = fragments[0]
    img = visualize_fragment(current, size=size)
    images.append(('Исходный: ' + current.name, img))

    for i, (frag, conn) in enumerate(zip(fragments[1:], connections), 1):
        my_point, other_point, bond_order = conn
        current = current.connect_to(frag, my_point, other_point, bond_order)

        img = visualize_fragment(current, size=size)
        title = f"Шаг {i}: +{frag.name} → {current.heavy_formula}, IHD={current.ihd}"
        images.append((title, img))

    return images


def print_molecule_info(mol: Molecule, index: int = None):
    """Вывести информацию о молекуле"""
    if index is not None:
        print(f"\n{'='*60}")
        print(f"СТРУКТУРА #{index}")
        print(f"{'='*60}")

    print(f"\nМолекула: {mol.get_formula()}")
    print(f"Атомов: {len(mol.atoms)}")
    print(f"Связей: {len(mol.edges)}")
    print(f"IHD: {mol.calculate_IHD()}")
    print(f"Связна: {'Да' if mol.is_connected() else 'Нет'}")

    print(f"\nАтомы (первые 10):")
    for i, atom in enumerate(mol.atoms[:10]):
        print(f"  {atom}")
    if len(mol.atoms) > 10:
        print(f"  ... и еще {len(mol.atoms) - 10} атомов")

    print(f"\nСвязи (первые 10):")
    bond_symbols = {1: '-', 2: '=', 3: '≡'}
    for i, (a1, a2, order) in enumerate(mol.edges[:10]):
        symbol1 = mol.atoms[a1].symbol
        symbol2 = mol.atoms[a2].symbol
        bond = bond_symbols.get(order, '-')
        print(f"  {symbol1}{a1} {bond} {symbol2}{a2}")
    if len(mol.edges) > 10:
        print(f"  ... и еще {len(mol.edges) - 10} связей")


def visualize_with_rdkit(mol: Molecule):
    """Визуализация молекулы через RDKit"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from IPython.display import display

        # Создаем RDKit молекулу
        rdkit_mol = Chem.RWMol()

        # Добавляем атомы
        atom_map = {}
        for i, atom in enumerate(mol.atoms):
            rd_atom = Chem.Atom(atom.symbol)
            if atom.formal_charge != 0:
                rd_atom.SetFormalCharge(atom.formal_charge)
            atom_map[i] = rdkit_mol.AddAtom(rd_atom)

        # Добавляем связи
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE
        }

        for a1, a2, order in mol.edges:
            rdkit_mol.AddBond(
                atom_map[a1],
                atom_map[a2],
                bond_types.get(order, Chem.BondType.SINGLE)
            )

        # Конвертируем в Mol
        final_mol = rdkit_mol.GetMol()

        # Оптимизируем геометрию
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(final_mol)

        # Отображаем
        img = Draw.MolToImage(final_mol, size=(400, 300))
        display(img)

        return final_mol

    except ImportError:
        print("⚠️ RDKit не установлен. Используйте: !pip install rdkit")
        return None
    except Exception as e:
        print(f"⚠️ Ошибка визуализации: {e}")
        return None



# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DELTA_CD3   = 17.03448   # Da: сдвиг m/z при замене COOH -> COOCD3
DELTA_CD3CO = 45.02939   # Da: сдвиг m/z при замене OH  -> OCOCD3


# ===========================================================================
# Загрузка спектров
# ===========================================================================

import logging

logger = logging.getLogger(__name__)

def load_spectrum(
    path,
    mapper=None,
    sep=",",
    mass_min=200.0,
    mass_max=700.0,
    metadata=None,
):
    """Загрузка спектра из CSV.

    Генерирует ValueError/KeyError при проблемах.
    GUI-слой решает, как эти ошибки показывать пользователю.
    """

    _sep = sep or ","

    try:
        df = pd.read_csv(path, sep=_sep, encoding="utf-8")
    except Exception as e:
        # Логируем на уровне core для разработчика
        logger.exception("Ошибка чтения CSV-файла %r", path)
        # Поднимаем дальше осмысленное исключение
        raise ValueError(f"Не удалось прочитать CSV-файл '{path}': {e}") from e

    df.columns = [c.strip() for c in df.columns]

    _default_mapper = {
        "m/z": "mass",
        "M/Z": "mass",
        "mz": "mass",
        "Intensity": "intensity",
        "I": "intensity",
    }

    final_mapper = _default_mapper.copy()
    if mapper:
        final_mapper.update(mapper)

    df = df.rename(columns=final_mapper)

    logger.debug(
        "Файл %r: колонки после rename: %r",
        path,
        df.columns.tolist(),
    )

    required = ["mass", "intensity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Колонки {missing} не найдены после переименования. "
            f"Доступные: {df.columns.tolist()}"
        )

    df = df[["mass", "intensity"]].copy()

    df = df[(df["mass"] >= mass_min) & (df["mass"] <= mass_max)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"Для файла '{path}' не найдено ни одного пика "
            f"в диапазоне {mass_min}–{mass_max} Da"
        )

    sp = Spectrum(table=df, metadata=metadata)
    return sp
# ===========================================================================
# Шумоподавление
# ===========================================================================

def denoise(
    spec,
    *,
    force=1.5,
    intensity=None,
    quantile=None,
):
    """
    Удалить шум из спектра (обёртка вокруг Spectrum.noise_filter()).

    Приоритет параметров:
        1. intensity  — жёсткий абсолютный порог;
        2. quantile   — нижний квантиль (0-1);
        3. force      — авто-детекция уровня шума, множитель (по умолчанию 1.5).
    """
    return spec.noise_filter(force=force, intensity=intensity, quantile=quantile)


# ===========================================================================
# ЭТАП 2b: Назначение брутто-формул
# ===========================================================================

DEFAULT_BRUTTO_DICT = {
    'C': (4, 50),
    'H': (4, 100),
    'O': (0, 25),
    'N': (0, 2),
}


def assign_formulas(
    src,
    *,
    brutto_dict=None,
    rel_error=0.5,
    sign='-',
    mass_min=None,
    mass_max=None,
):
    """
    Назначить брутто-формулы пикам исходного спектра.

    После назначения добавляются столбцы:
        brutto     — строковая формула (например 'C15H24O12')
        calc_mass  — теоретическая масса

    Параметры
    ---------
    src : Spectrum
    brutto_dict : dict, optional
        Элементы и диапазоны их количеств. По умолчанию DEFAULT_BRUTTO_DICT.
    rel_error : float
        Допустимая погрешность назначения (ppm). Типично 0.5 ppm для Orbitrap.
    sign : {'-', '+', '0'}
        Режим ионизации.
    mass_min, mass_max : float, optional
        Ограничение диапазона масс при назначении.
    """
    if rel_error < 0:
        rel_error = abs(rel_error)
        warnings.warn("Relative error is negative")
    if mass_min > mass_max:
        (mass_min, mass_max) = (mass_max, mass_min)
        warnings.warn("Mass_max is less than mass_min")

    if not isinstance(src, Spectrum): raise TypeError(f'Некорректный формат файла {src}')
    if brutto_dict is None:
        brutto_dict = DEFAULT_BRUTTO_DICT
    elif not isinstance(brutto_dict, dict):
        raise TypeError("brutto_dict должен быть dict с диапазонами по элементам")
    for el, bounds in brutto_dict.items():
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            raise ValueError(f"Для элемента {el!r} ожидается (min, max), получено {bounds!r}")

    src = src.assign(
        brutto_dict=brutto_dict,
        rel_error=rel_error,
        sign=sign,
        mass_min=mass_min,
        mass_max=mass_max,
    )
    if "assign" not in src.table.columns:
        raise RuntimeError(
            "После вызова src.assign в таблице src.table нет колонки 'assign'"
        )

    assign_col = src.table["assign"]
    if assign_col.dtype != bool:
        raise TypeError(f"Ожидается булевый столбец 'assign', получен dtype={assign_col.dtype}")

    n_assigned = int(assign_col.sum())
    if n_assigned > 0:
        src = src.calc_mass()
        src = src.brutto()
    elif n_assigned == 0: warnings.warn("Ни одной брутто-формулы не назначено (assign == False для всех пиков)")
    else:
        if hasattr(src, "calc_mass"):
            src.calc_mass()
        if hasattr(src, "brutto"):
            src.brutto()

    return src


# ===========================================================================
# Поиск серий
# ===========================================================================

def _find_peak(mz_array, target_mz, ppm_tol):
    """
    Найти индекс ближайшего пика в отсортированном mz_array к target_mz.
    Возвращает int или None.
    """
    idx = np.searchsorted(mz_array, target_mz)
    best = None
    best_err = ppm_tol
    for i in (idx - 1, idx):
        if 0 <= i < len(mz_array):
            err_ppm = abs(mz_array[i] - target_mz) / target_mz * 1e6
            if err_ppm < best_err:
                best_err = err_ppm
                best = i
    return best


def find_series(
    src,
    deriv,
    delta,
    ppm_tol=5.0,
    max_groups=20,
    allow_gaps=True,
    min_series_length=1,
):
    """
    Найти серии дейтериационных пиков.

    Для каждого назначенного пика m_0 ищет цепочку:
        m_0 + 1*delta,  m_0 + 2*delta,  ...,  m_0 + n*delta
    в дериватизированном спектре.

    Правило определения длины серии
    ---------------------------------
    Длина серии = последний НАЙДЕННЫЙ шаг (1-based).
    Если между первым и последним шагами есть пропуски,
    они фиксируются в поле missing.
    Логика: "видим 1,2,3,5 -> считаем серию длиной 5".

    Параметры
    ---------
    src : Spectrum
        Исходный спектр с назначенными формулами.
    deriv : Spectrum
        Спектр дериватизированного образца.
    delta : float
        Ожидаемый сдвиг m/z на одну функциональную группу (Da).
    ppm_tol : float
        Допустимая погрешность совпадения масс (ppm).
    max_groups : int
        Максимально возможное число функциональных групп на молекулу.
    allow_gaps : bool
        True  — продолжать поиск при пропуске (рекомендуется).
        False — обрывать серию на первом пропуске.
    min_series_length : int
        Минимальная длина серии для включения в вывод.

    Возвращает
    ----------
    DataFrame:
        mass_src    — m/z пика в исходном спектре
        brutto      — назначенная брутто-формула
        n_groups    — длина серии (по последнему найденному шагу)
        steps_found — список найденных шагов (1-based)
        missing     — список пропущенных шагов ВНУТРИ серии
        series_mz   — список m/z для шагов 1..n_groups (None = пропуск)
    """
    if ppm_tol <= 0:
        raise ValueError(f"ppm_tol должно быть > 0, получено {ppm_tol}")
    if max_groups < 1 or min_series_length < 1:
        raise ValueError(
            f"max_groups ({max_groups}) и min_series_length ({min_series_length}) "
            "должны быть >= 1"
        )
    required_src = ['brutto', 'mass', 'assign']
    missing_src = [c for c in required_src if c not in src.table.columns]
    if missing_src:
        raise ValueError(f"В src не хватает столбца {missing_src}")
    required_deriv = ['mass', 'intensity']
    missing_deriv = [c for c in required_deriv if c not in deriv.table.columns]
    if missing_deriv:
        raise ValueError(
            f"В deriv.table отсутствуют колонки {missing_deriv}. "
            "Файл дериватизированного спектра некорректен."
        )

    mz_deriv = deriv.table['mass'].values
    records  = []

    for _, row in src.table.iterrows():
        if not row.get('assign', False):
            continue

        m0          = row['mass']
        found_steps = []
        series_mz   = []

        for step in range(1, max_groups + 1):
            target = m0 + step * delta
            idx = _find_peak(mz_deriv, target, ppm_tol)

            if idx is not None:
                found_steps.append(step)
                series_mz.append(float(mz_deriv[idx]))
            else:
                series_mz.append(None)
                if not allow_gaps and found_steps:
                    series_mz = series_mz[:step]
                    break

        if not found_steps:
            n_groups      = 0
            missing_steps = []
            trimmed       = []
        else:
            n_groups      = max(found_steps)
            all_steps     = set(range(1, n_groups + 1))
            missing_steps = sorted(all_steps - set(found_steps))
            trimmed       = series_mz[:n_groups]

        if n_groups >= min_series_length:
            records.append({
                'mass_src':    m0,
                'brutto':      row.get('brutto', ''),
                'n_groups':    n_groups,
                'steps_found': found_steps,
                'missing':     missing_steps,
                'series_mz':   trimmed,
            })

    return pd.DataFrame(records)


# ===========================================================================
# Сборка итоговой таблицы
# ===========================================================================

def build_result_table(src, df_dmet, df_dacet):
    """
    Собрать итоговую таблицу с числом -COOH и -OH для каждой брутто-формулы.

    Логика:
        N_COOH     = n_groups из df_dmet  (серия CD3,   delta = 17.034 Da)
        N_OH_total = n_groups из df_dacet (серия CD3CO, delta = 45.029 Da)
        N_OH       = N_OH_total - N_COOH  (чистые спиртовые ОН)

    Параметры
    ---------
    src : Spectrum
    df_dmet : DataFrame — результат find_series() для дейтерометилирования.
    df_dacet : DataFrame — результат find_series() для дейтероацилирования.

    Возвращает
    ----------
    DataFrame: mass, intensity, brutto, N_COOH, N_OH_total, N_OH,
               missing_dmet, missing_dacet
    """
    base = (
        src.table
        .loc[src.table.get('assign', pd.Series(False, index=src.table.index)) == True]
        [['mass', 'intensity', 'brutto']]
        .copy()
        .reset_index(drop=True)
    )
    base['mass_key'] = base['mass'].round(4)

    def _enrich(df, prefix):
        if df.empty:
            return pd.DataFrame(columns=['mass_key', f'n_{prefix}', f'missing_{prefix}'])
        tmp = df[['mass_src', 'n_groups', 'missing']].copy()
        tmp['mass_key'] = tmp['mass_src'].round(4)
        return tmp.rename(columns={
            'n_groups': f'n_{prefix}',
            'missing':  f'missing_{prefix}',
        })[['mass_key', f'n_{prefix}', f'missing_{prefix}']]

    result = (
        base
        .merge(_enrich(df_dmet,  'dmet'),  on='mass_key', how='left')
        .merge(_enrich(df_dacet, 'dacet'), on='mass_key', how='left')
    )

    result['n_dmet']  = result['n_dmet'].fillna(0).astype(int)
    result['n_dacet'] = result['n_dacet'].fillna(0).astype(int)
    result['N_COOH']     = result['n_dmet']
    result['N_OH_total'] = result['n_dacet']
    result['N_OH'] = result['n_dacet']

    impossible = result[result['N_OH_total'] < result['N_COOH']]
    if not impossible.empty:
        warnings.warn(
            f"{len(impossible)} пик(ов): N_OH_total < N_COOH. "
            "Возможна ошибка назначения серий или частичная дериватизация."
        )

    return result[[
        'mass', 'intensity', 'brutto',
        'N_COOH', 'N_OH_total', 'N_OH',
        'missing_dmet', 'missing_dacet',
    ]].sort_values('mass').reset_index(drop=True)


# ===========================================================================
# Визуализация серий с пропущенными пиками
# ===========================================================================

def visualize_series(
    src,
    deriv,
    df_series,
    delta,
    label="series",
    max_rows=15,
    figsize_per_row=(12, 1.4),
    ppm_tol=5.0,
    save_path=None,
):
    """
    Визуализировать серии с пропущенными пиками.

    Для каждого соединения строится лесенка ожидаемых пиков:
        синий      — исходный пик (m_0)
        зелёный    — найденный пик серии
        красный -- — пропущенный ожидаемый пик

    Параметры
    ---------
    src : Spectrum
    deriv : Spectrum
    df_series : DataFrame — результат find_series().
    delta : float — шаг серии (Da).
    label : str — подпись в заголовке.
    max_rows : int — максимальное число соединений для отображения.
    figsize_per_row : tuple — (ширина, высота) одной строки.
    ppm_tol : float — допуск поиска (ppm).
    save_path : str, optional — путь для сохранения рисунка.
    """
    if df_series.empty:
        print(f"[{label}] Серии не найдены.")
        return

    has_missing = df_series[df_series['missing'].apply(len) > 0]
    display_df  = has_missing.head(max_rows)

    if display_df.empty:
        print(f"[{label}] Пропущенных пиков в сериях нет.")
        return

    n_rows = len(display_df)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows + 1.5),
        squeeze=False,
    )
    fig.suptitle(
        f"Серии {label} с пропущенными пиками "
        f"(delta_m = {delta:.5f} Da, допуск {ppm_tol} ppm)",
        fontsize=11, fontweight="bold",
    )

    mz_src    = src.table['mass'].values
    int_src   = src.table['intensity'].values
    mz_deriv  = deriv.table['mass'].values
    int_deriv = deriv.table['intensity'].values

    for ax_idx, (_, row) in enumerate(display_df.iterrows()):
        ax       = axes[ax_idx][0]
        m0       = row['mass_src']
        n_groups = row['n_groups']
        missing  = set(row['missing'])
        series   = row['series_mz']

        idx_s = _find_peak(mz_src, m0, ppm_tol * 10)
        i0    = float(int_src[idx_s]) if idx_s is not None else 1.0

        max_i = i0
        for mz_step in series:
            if mz_step is not None:
                idx_d = _find_peak(mz_deriv, mz_step, ppm_tol * 2)
                if idx_d is not None:
                    max_i = max(max_i, float(int_deriv[idx_d]))

        bar_w = delta * 0.08
        ax.bar(m0, i0, width=bar_w, color='steelblue', alpha=0.85)

        for step, mz_step in enumerate(series, start=1):
            expected = m0 + step * delta
            if step in missing or mz_step is None:
                ax.axvline(x=expected, color='crimson',
                           linestyle='--', linewidth=1.0, alpha=0.75)
                ax.text(expected, max_i * 0.55, f"n={step}",
                        color='crimson', fontsize=7, ha='center', va='bottom')
            else:
                idx_d = _find_peak(mz_deriv, float(mz_step), ppm_tol * 2)
                i_step = float(int_deriv[idx_d]) if idx_d is not None else max_i * 0.1
                ax.bar(mz_step, i_step, width=bar_w, color='forestgreen', alpha=0.8)
                ax.text(mz_step, i_step + max_i * 0.02, f"n={step}",
                        color='darkgreen', fontsize=7, ha='center', va='bottom')

        ax.set_xlim(m0 - delta * 0.5, m0 + (n_groups + 1) * delta)
        ax.set_ylim(0, max_i * 1.25)
        ax.set_ylabel('I', fontsize=8)
        ax.set_title(
            f"{row['brutto']}   m/z={m0:.4f}   "
            f"серия 1..{n_groups}   пропущено: {sorted(missing)}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

    fig.legend(
        handles=[
            mpatches.Patch(color='steelblue',   label='Исходный пик'),
            mpatches.Patch(color='forestgreen', label='Найденный пик серии'),
            mpatches.Patch(color='crimson',     label='Пропущенный пик (ожидаемая позиция)'),
        ],
        loc='lower center', ncol=3, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 0),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[{label}] График сохранён: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ===========================================================================
# ГЛАВНАЯ ФУНКЦИЯ — полный пайплайн
# ===========================================================================

def run_pipeline(
    src_path,
    dmet_path,
    dacet_path,
    *,
    # Загрузка
    sep=",",
    mass_min=200.0,
    mass_max=700.0,
    # Шумоподавление
    noise_force=1.5,
    noise_intensity=None,
    noise_quantile=None,
    # Назначение формул
    brutto_dict=None,
    rel_error=0.5,
    sign='-',
    # Поиск серий
    ppm_tol=5.0,
    max_groups=20,
    allow_gaps=True,
    # Визуализация
    visualize=True,
    save_dmet=None,
    save_dacet=None,
    # Выходной файл
    output_csv=None,
):
    """
    Полный пайплайн определения числа -COOH и -OH групп.

    Параметры
    ---------
    src_path, dmet_path, dacet_path : str | Path
        Пути к CSV-файлам трёх спектров.
    src_mapper, deriv_mapper : dict, optional
        Словари переименования столбцов {'m/z': 'mass', 'I': 'intensity'}.
    sep : str
        Разделитель полей CSV.
    mass_min, mass_max : float
        Рабочий диапазон масс (Da).
    noise_force / noise_intensity / noise_quantile
        Параметры шумоподавления (см. denoise()).
    brutto_dict : dict, optional
        Элементы и диапазоны для назначения формул.
    rel_error : float
        Допустимая ошибка назначения (ppm).
    sign : {'-', '+', '0'}
        Режим ионизации.
    ppm_tol : float
        Допуск поиска пика в серии (ppm). Рекомендуется 3-5 ppm.
    max_groups : int
        Максимальное ожидаемое число групп на молекулу.
    allow_gaps : bool
        Разрешать ли пропуски внутри серии (рекомендуется True).
    visualize : bool
        Строить ли графики для серий с пропущенными пиками.
    save_dmet, save_dacet : str, optional
        Пути для сохранения графиков (PNG/PDF).
    output_csv : str, optional
        Сохранить итоговую таблицу в CSV.

    Возвращает
    ----------
    pd.DataFrame — итоговая таблица.
    """
    print('=' * 60)
    print('ШАГ 1: Загрузка спектров')
    print('=' * 60)
    _mapper = {"mass": "mass", "intensity": "intensity"}
    src   = load_spectrum(src_path,   mapper = _mapper, sep=sep,
                          mass_min=mass_min, mass_max=mass_max, metadata={'name': 'src'})
    dmet  = load_spectrum(dmet_path,  mapper = _mapper, sep=sep,
                          mass_min=mass_min, mass_max=mass_max, metadata={'name': 'dmet'})
    dacet = load_spectrum(dacet_path, mapper = _mapper, sep=sep,
                          mass_min=mass_min, mass_max=mass_max, metadata={'name': 'dacet'})
    print(f"  Загружено пиков:  src={len(src)},  dmet={len(dmet)},  dacet={len(dacet)}")

    print()
    print('=' * 60)
    print('ШАГ 2a: Шумоподавление')
    print('=' * 60)
    src   = denoise(src,   force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
    dmet  = denoise(dmet,  force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
    dacet = denoise(dacet, force=noise_force, intensity=noise_intensity, quantile=noise_quantile)
    print(f"  После шумоподавления: src={len(src)},  dmet={len(dmet)},  dacet={len(dacet)}")

    print()
    print('=' * 60)
    print('ШАГ 2b: Назначение брутто-формул исходному спектру')
    print('=' * 60)
    src = assign_formulas(src, brutto_dict=brutto_dict, rel_error=rel_error,
                          sign=sign, mass_min=mass_min, mass_max=mass_max)
    n_assigned = int(src.table.get('assign', pd.Series(dtype=bool)).sum())
    print(f"  Назначено формул: {n_assigned} из {len(src)} пиков")

    print()
    print('=' * 60)
    print('ШАГ 3: Серии дейтерометилирования (-> N_COOH)')
    print('=' * 60)
    df_dmet = find_series(src, dmet, delta=DELTA_CD3,
                          ppm_tol=ppm_tol, max_groups=max_groups, allow_gaps=allow_gaps)
    print(f"  Соединений с сериями CD3: {len(df_dmet)}")
    if not df_dmet.empty:
        print(f"  Макс. N_COOH = {df_dmet['n_groups'].max()}")
        ng = df_dmet['missing'].apply(len).sum()
        if ng:
            print(f"  ВНИМАНИЕ: Внутренних пропусков в сериях: {ng}")

    print()
    print('=' * 60)
    print('ШАГ 4: Серии дейтероацилирования (-> N_OH_total)')
    print('=' * 60)
    df_dacet = find_series(src, dacet, delta=DELTA_CD3CO,
                           ppm_tol=ppm_tol, max_groups=max_groups, allow_gaps=allow_gaps)
    print(f"  Соединений с сериями CD3CO: {len(df_dacet)}")
    if not df_dacet.empty:
        print(f"  Макс. N_OH_total = {df_dacet['n_groups'].max()}")
        ng2 = df_dacet['missing'].apply(len).sum()
        if ng2:
            print(f"  ВНИМАНИЕ: Внутренних пропусков в сериях: {ng2}")

    print()
    print('=' * 60)
    print('ШАГ 5: Итоговая таблица N_COOH / N_OH')
    print('=' * 60)
    result = build_result_table(src, df_dmet, df_dacet)
    print(f"  Строк в таблице: {len(result)}")
    print(f"  Соединений с N_COOH > 0: {(result['N_COOH'] > 0).sum()}")
    print(f"  Соединений с N_OH   > 0: {(result['N_OH']   > 0).sum()}")

    if visualize:
        print()
        print('=' * 60)
        print('ШАГ 6: Визуализация пропущенных пиков')
        print('=' * 60)
        visualize_series(src, dmet, df_dmet,
                         delta=DELTA_CD3, label="дейтерометилирования",
                         ppm_tol=ppm_tol, save_path=save_dmet)
        visualize_series(src, dacet, df_dacet,
                         delta=DELTA_CD3CO, label="дейтероацилирования",
                         ppm_tol=ppm_tol, save_path=save_dacet)

    if output_csv:
        result.to_csv(output_csv, index=False, sep=';', encoding='utf-8-sig')
        print(f"\nИтоговая таблица сохранена: {output_csv}")

    return result


# ===========================================================================
# Пример использования
# ===========================================================================

if __name__ == '__main__':
    result = run_pipeline(
        src_path   =r"/testdata/src_spectrum.csv",
        dmet_path  =r"/testdata/dmet_spectrum.csv",
        dacet_path =r"/testdata/dacet_spectrum.csv",

        # --- Переименование столбцов (если нужно) ---
        # src_mapper   = {'m/z': 'mass', 'Intensity': 'intensity'},
        # deriv_mapper = {'m/z': 'mass', 'Intensity': 'intensity'},

        sep        = "\t",          # tab-разделитель (типично для FreeStyle export)
        mass_min   = 200.0,
        mass_max   = 700.0,

        # --- Шумоподавление ---
        noise_force = 1.5,

        # --- Назначение формул ---
        brutto_dict = {'C':(4,50), 'H':(4,100), 'O':(0,25), 'N':(0,2)},
        rel_error   = 0.5,           # ppm для Orbitrap Elite
        sign        = '-',

        # --- Поиск серий ---
        ppm_tol    = 5.0,
        max_groups = 20,
        allow_gaps = True,

        # --- Вывод ---
        visualize  = True,
        save_dmet  = "gaps_dmet.png",
        save_dacet = "gaps_dacet.png",
        output_csv = "result_table.csv",
    )

    print("\nПервые 10 строк результата:")
    print(result.head(10).to_string(index=False))


