"""Dictionary with physical dimensions"""
from tdeptools.keys import keys

time, atom, cart = "time", "atom", "cart"
dimensions = {
    keys.cell: (cart, cart),
    keys.positions: (atom, cart),
    keys.positions_cartesian: (atom, cart),
    keys.forces: (atom, cart),
    keys.stress: (cart, cart),
    keys.dielectric_tensor: (cart, cart),
    keys.born_charges: (atom, cart, cart),
}
