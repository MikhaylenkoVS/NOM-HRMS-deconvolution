"""Bridge between the internal molecule model and RDKit.

Provides converters and visualization helpers that turn the project's own
``Molecule`` / ``MoleculeFragment`` objects into RDKit ``Mol`` objects and
2D depictions. RDKit (and IPython, for inline display) is imported lazily so
the rest of the pipeline can run without it installed.
"""
from src.core.molecule import Molecule

def to_rdkit_mol(fragment: Molecule):
    """Convert an internal molecule/fragment to a sanitized RDKit ``Mol``.

    Parameters
    ----------
    fragment : Molecule or MoleculeFragment
        Object exposing ``atoms`` (element symbols) and ``bonds`` as
        ``(i, j, order)`` triples.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule with explicit hydrogens and computed 2D coordinates.

    Notes
    -----
    Sanitization falls back to a reduced operation set if full sanitization
    fails, so partially specified fragments can still be drawn.
    """
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


def visualize_fragment(fragment: Molecule,
                       highlight_attachment_points: bool = True,
                       size: tuple = (400, 300)):
    """Render a single fragment to a PIL image with RDKit.

    Parameters
    ----------
    fragment : Molecule or MoleculeFragment
        Fragment to draw.
    highlight_attachment_points : bool, optional
        If ``True``, highlight the fragment's free attachment points.
        Default ``True``.
    size : tuple of (int, int), optional
        Image size in pixels ``(width, height)``. Default ``(400, 300)``.

    Returns
    -------
    PIL.Image.Image
        Rendered depiction of the fragment.
    """
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
    """Render several fragments as a labelled grid image.

    Parameters
    ----------
    fragments : list of Molecule or MoleculeFragment
        Fragments to draw.
    names : list of str, optional
        Display names, one per fragment. Defaults to each fragment's
        ``name`` attribute.
    mols_per_row : int, optional
        Number of depictions per row. Default 3.
    subImgSize : tuple of (int, int), optional
        Size in pixels ``(width, height)`` of each cell. Default
        ``(300, 250)``.

    Returns
    -------
    PIL.Image.Image
        Grid image whose legends show the name, heavy-atom formula and
        IHD of each fragment.
    """
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
    """Draw the stepwise assembly of a molecule from its fragments.

    Starting from the first fragment, each subsequent fragment is joined
    through the specified attachment points and the growing structure is
    rendered after every step.

    Parameters
    ----------
    fragments : list of Molecule or MoleculeFragment
        Fragments to connect in order; ``fragments[0]`` is the starting
        skeleton.
    connections : list of tuple
        One ``(my_point, other_point, bond_order)`` triple per fragment
        after the first, describing how it attaches to the current
        structure.
    size : tuple of (int, int), optional
        Image size in pixels ``(width, height)``. Default ``(400, 300)``.

    Returns
    -------
    list of tuple
        ``(title, PIL.Image.Image)`` pairs, one for the starting fragment
        and one for each connection step.
    """
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
    """Print a human-readable summary of a molecule to stdout.

    Parameters
    ----------
    mol : Molecule
        Molecule to describe.
    index : int, optional
        Structure number shown in a header banner. If ``None`` no header
        is printed.

    Returns
    -------
    None
        Output (formula, atom/bond counts, IHD, connectivity, and the
        first ten atoms and bonds) is written to stdout.
    """
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
    """Render a molecule inline with RDKit and IPython display.

    Builds an RDKit ``Mol`` from the internal representation (preserving
    formal charges), computes 2D coordinates and displays the depiction,
    e.g. inside a Jupyter notebook.

    Parameters
    ----------
    mol : Molecule
        Molecule to visualize.

    Returns
    -------
    rdkit.Chem.Mol or None
        The constructed RDKit molecule, or ``None`` if RDKit/IPython are
        not installed or rendering fails.
    """
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
