from enum import Enum
from typing import List

ELEMENT_DATA = {
    "H": {"atomic_number": 1, "valence": 1},
    "C": {"atomic_number": 6, "valence": 4},
    "N": {"atomic_number": 7, "valence": 3, "valence_charged": {-1: 2, 0: 3, 1: 4}},
    "O": {"atomic_number": 8, "valence": 2, "valence_charged": {-1: 1, 0: 2, 1: 3}},
    "F": {"atomic_number": 9, "valence": 1},
    "P": {"atomic_number": 15, "valence": 5, "valence_charged": {0: 5, 1: 4}},
    "S": {"atomic_number": 16, "valence": 6, "valence_charged": {0: 6, 1: 3, -1: 1}},
    "Cl": {"atomic_number": 17, "valence": 1},
    "Br": {"atomic_number": 35, "valence": 1},
    "I": {"atomic_number": 53, "valence": 1},
}


class Hybridization(Enum):
    """Atomic hybridization state.

    Enumerates the carbon hybridization states inferred from local bonding.

    Attributes
    ----------
    SP3 : str
        Tetrahedral carbon, only single bonds.
    SP2 : str
        Trigonal carbon, one double bond or aromatic.
    SP : str
        Linear carbon, one triple bond.
    UNKNOWN : str
        Hybridization not yet determined.
    """

    SP3 = "sp3"
    SP2 = "sp2"
    SP = "sp"
    UNKNOWN = "unknown"


class Atom:
    """A single atom within a molecular graph.

    Tracks the element, its bonds, and the consumed valence so that the
    molecule builder can enforce valence limits while assembling structures.

    Parameters
    ----------
    symbol : str
        Chemical element symbol. Must be a key of ``ELEMENT_DATA``
        (H, C, N, O, F, P, S, Cl, Br, I).
    number : int
        Unique index of the atom within its molecule.
    formal_charge : int, optional
        Formal charge on the atom. For elements with charge-dependent
        valence (N, O, P, S) the effective valence is selected accordingly.
        Default is 0.

    Attributes
    ----------
    valence : int
        Maximum bond order the atom can accommodate (charge-adjusted).
    connections : list of int
        Indices of bonded neighbour atoms.
    bond_orders : list of int
        Bond order for each entry in ``connections`` (1, 2 or 3).
    used_valence : int
        Sum of bond orders currently attached to the atom.
    hybridization : Hybridization
        Current hybridization state (auto-updated for carbon).
    is_aromatic : bool
        Whether the atom belongs to an aromatic system.

    Raises
    ------
    ValueError
        If ``symbol`` is not a supported element.
    """

    def __init__(self, symbol: str, number: int, formal_charge: int = 0):
        if symbol not in ELEMENT_DATA:
            raise ValueError(f"Неподдерживаемый элемент: {symbol}")

        self.symbol = symbol
        self.number = number
        self.formal_charge = formal_charge

        element = ELEMENT_DATA[symbol]
        self.atomic_number = element["atomic_number"]

        if "valence_charged" in element and formal_charge in element["valence_charged"]:
            self.valence = element["valence_charged"][formal_charge]
        else:
            self.valence = element["valence"]

        self.connections: List[int] = []
        self.bond_orders: List[int] = []
        self.used_valence = 0
        self.hybridization = Hybridization.UNKNOWN
        self.is_aromatic = False

    def add_bond(self, atom_number: int, bond_order: int = 1) -> bool:
        """Add a bond to another atom, respecting the valence limit.

        Parameters
        ----------
        atom_number : int
            Index of the neighbour atom to bond to.
        bond_order : int, optional
            Bond order to create (1 = single, 2 = double, 3 = triple).
            Default is 1.

        Returns
        -------
        bool
            ``True`` if the bond was added; ``False`` if the bond already
            exists or would exceed the atom's available valence.
        """
        if atom_number in self.connections:
            return False  # bond already exists
        if self.used_valence + bond_order > self.valence:
            return False
        self.connections.append(atom_number)
        self.bond_orders.append(bond_order)
        self.used_valence += bond_order
        self._update_hybridization()
        return True

    def _update_hybridization(self):
        """Infer and store the carbon hybridization from current bonds.

        Notes
        -----
        Only carbon atoms are classified: a triple bond implies ``SP``, a
        double bond or aromatic flag implies ``SP2``, otherwise ``SP3``.
        For non-carbon atoms the method returns without changes.
        """
        if self.symbol != "C":
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
        """Return the bond order to a given neighbour atom.

        Parameters
        ----------
        atom_number : int
            Index of the neighbour atom.

        Returns
        -------
        int
            Bond order to the neighbour (1, 2 or 3), or 0 if the two atoms
            are not bonded.
        """
        try:
            idx = self.connections.index(atom_number)
            return self.bond_orders[idx]
        except ValueError:
            return 0

    def __repr__(self):
        charge_str = f"{self.formal_charge:+d}" if self.formal_charge != 0 else ""
        return f"Atom({self.symbol}{charge_str}, #{self.number}, val={self.used_valence}/{self.valence})"
