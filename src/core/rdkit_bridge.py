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
