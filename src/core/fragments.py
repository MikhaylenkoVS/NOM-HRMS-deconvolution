from typing import Dict, List, Tuple
from .atoms import Atom, Hybridization
import numpy as np

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
        adj = dict(list)
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