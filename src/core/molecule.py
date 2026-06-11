from collections import defaultdict
from typing import List, Tuple, Dict
from .atoms import Atom

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




