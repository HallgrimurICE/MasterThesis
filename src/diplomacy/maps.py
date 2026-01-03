from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .graph import nx
from .state import GameState
from .types import Power, Province, ProvinceType, Unit, UnitType


def build_graph(board: Dict[str, Province]) -> nx.Graph:  # type: ignore[override]
    graph = nx.Graph()  # type: ignore[assignment]
    for name, prov in board.items():
        graph.add_node(
            name,
            is_supply_center=prov.is_supply_center,
            province_type=prov.province_type,
        )
        for nbr in prov.neighbors:
            if name != nbr:
                graph.add_edge(name, nbr)
    return graph


def square_board() -> Dict[str, Province]:
    adjacency = {
        "A": {"B", "C", "D"},
        "B": {"A", "C", "D"},
        "C": {"A", "B", "D"},
        "D": {"A", "B", "C"},
    }
    home_lookup = {
        "A": Power("Aurora"),
        "B": Power("Borealis"),
        "C": Power("Crimson"),
        "D": None,
    }
    board: Dict[str, Province] = {}
    for name, neighbors in adjacency.items():
        board[name] = Province(
            name=name,
            neighbors=set(neighbors),
            is_supply_center=True,
            home_power=home_lookup.get(name),
            province_type=ProvinceType.LAND,
        )
    return board


def triangle_board() -> Dict[str, Province]:
    adjacency = {
        "RED": {"RB", "GR", "MID"},
        "BLUE": {"RB", "BG", "MID"},
        "GREEN": {"BG", "GR", "MID"},
        "RB": {"RED", "BLUE", "MID"},
        "BG": {"BLUE", "GREEN", "MID"},
        "GR": {"GREEN", "RED", "MID"},
        "MID": {"RED", "BLUE", "GREEN", "RB", "BG", "GR"},
    }
    home_lookup = {
        "RED": Power("Red"),
        "BLUE": Power("Blue"),
        "GREEN": Power("Green"),
    }
    supply_centers = {"RED", "BLUE", "GREEN", "MID"}
    board: Dict[str, Province] = {}
    for name, neighbors in adjacency.items():
        board[name] = Province(
            name=name,
            neighbors=set(neighbors),
            is_supply_center=name in supply_centers,
            home_power=home_lookup.get(name),
            province_type=ProvinceType.LAND,
        )
    return board


def standard_board() -> Dict[str, Province]:
    board: Dict[str, Province] = {
        'NAO': Province(
            name='NAO',
            neighbors={'CLY', 'MAO', 'IRI', 'LVP', 'NWG'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'MAO': Province(
            name='MAO',
            neighbors={'ENG', 'NAO', 'IRI'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'IRI': Province(
            name='IRI',
            neighbors={'NAO', 'MAO', 'LVP', 'ENG', 'WAL'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'ENG': Province(
            name='ENG',
            neighbors={'MAO', 'IRI', 'WAL', 'LON', 'NTH'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'CLY': Province(
            name='CLY',
            neighbors={'NWG', 'NAO', 'LVP'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'LVP': Province(
            name='LVP',
            neighbors={'CLY', 'NAO', 'IRI', 'WAL'},
            is_supply_center=True,
            home_power=Power('England'),
            province_type=ProvinceType.COAST,
        ),
        'WAL': Province(
            name='WAL',
            neighbors={'LVP', 'IRI', 'ENG', 'LON'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'LON': Province(
            name='LON',
            neighbors={'ENG', 'WAL', 'YOR', 'NTH'},
            is_supply_center=True,
            home_power=Power('England'),
            province_type=ProvinceType.COAST,
        ),
        'YOR': Province(
            name='YOR',
            neighbors={'LVP', 'LON', 'NTH', 'EDI'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'EDI': Province(
            name='EDI',
            neighbors={'YOR', 'LVP', 'NTH', 'NWG', 'CLY'},
            is_supply_center=True,
            home_power=Power('England'),
            province_type=ProvinceType.COAST,
        ),
        'NWG': Province(
            name='NWG',
            neighbors={'CLY', 'NAO', 'NTH', 'NWY', 'BAR'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'BAR': Province(
            name='BAR',
            neighbors={'NWG', 'NWY', 'STP'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'NTH': Province(
            name='NTH',
            neighbors={'EDI', 'NWG', 'YOR', 'LON', 'ENG', 'DEN', 'SKA', 'NWY', 'HEL'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'NWY': Province(
            name='NWY',
            neighbors={'NWG', 'NTH', 'SKA', 'SWE', 'FIN', 'STP', 'BAR'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'SKA': Province(
            name='SKA',
            neighbors={'NTH', 'DEN', 'NWY', 'SWE'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'DEN': Province(
            name='DEN',
            neighbors={'NTH', 'SKA', 'SWE', 'BAL', 'HEL'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'HEL': Province(
            name='HEL',
            neighbors={'NTH', 'DEN', 'BAL'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'SWE': Province(
            name='SWE',
            neighbors={'SKA', 'DEN', 'BAL', 'BOT', 'FIN', 'NWY'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'BAL': Province(
            name='BAL',
            neighbors={'DEN', 'SWE', 'BOT', 'LVN', 'PRU', 'HEL'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'BOT': Province(
            name='BOT',
            neighbors={'BAL', 'SWE', 'FIN', 'LVN', 'STP'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'FIN': Province(
            name='FIN',
            neighbors={'NWY', 'SWE', 'BOT', 'STP'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'STP': Province(
            name='STP',
            neighbors={'BAR', 'NWY', 'FIN', 'BOT', 'LVN', 'MOS'},
            is_supply_center=True,
            home_power=Power('Russia'),
            province_type=ProvinceType.COAST,
        ),
        'LVN': Province(
            name='LVN',
            neighbors={'BAL', 'BOT', 'STP', 'MOS', 'WAR', 'PRU'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'PRU': Province(
            name='PRU',
            neighbors={'BAL', 'LVN', 'WAR'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'MOS': Province(
            name='MOS',
            neighbors={'STP', 'LVN', 'WAR', 'SEV'},
            is_supply_center=True,
            home_power=Power('Russia'),
            province_type=ProvinceType.LAND,
        ),
        'WAR': Province(
            name='WAR',
            neighbors={'LVN', 'PRU', 'MOS'},
            is_supply_center=True,
            home_power=Power('Russia'),
            province_type=ProvinceType.LAND,
        ),
        'SEV': Province(
            name='SEV',
            neighbors={'MOS', 'UKR', 'RUM', 'BLA', 'ARM'},
            is_supply_center=True,
            home_power=Power('Russia'),
            province_type=ProvinceType.LAND,
        ),
        'BLA': Province(
            name='BLA',
            neighbors={'SEV', 'RUM', 'BUL', 'CON', 'ANK', 'ARM'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'UKR': Province(
            name='UKR',
            neighbors={'SEV', 'MOS', 'WAR', 'GAL', 'RUM'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'GAL': Province(
            name='GAL',
            neighbors={'UKR', 'WAR', 'SIL', 'BOH', 'VIE', 'BUD', 'RUM'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'SIL': Province(
            name='SIL',
            neighbors={'WAR', 'GAL', 'BOH', 'PRU', 'BER', 'MUN'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'KIE': Province(
            name='KIE',
            neighbors={'DEN', 'HEL', 'HOL', 'BER', 'MUN', 'BAL', 'RUH'},
            is_supply_center=True,
            home_power=Power('Germany'),
            province_type=ProvinceType.COAST,
        ),
        'BER': Province(
            name='BER',
            neighbors={'KIE', 'PRU', 'SIL', 'MUN', 'BAL'},
            is_supply_center=True,
            home_power=Power('Germany'),
            province_type=ProvinceType.COAST,
        ),
        'MUN': Province(
            name='MUN',
            neighbors={'KIE', 'BER', 'SIL', 'BOH', 'TYR', 'RUH', 'BUR'},
            is_supply_center=True,
            home_power=Power('Germany'),
            province_type=ProvinceType.LAND,
        ),
        'BOH': Province(
            name='BOH',
            neighbors={'MUN', 'SIL', 'GAL', 'VIE', 'TYR'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'TYR': Province(
            name='TYR',
            neighbors={'MUN', 'BOH', 'VIE', 'PIE', 'VEN', 'TRI'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'VIE': Province(
            name='VIE',
            neighbors={'BOH', 'GAL', 'BUD', 'TRI', 'TYR'},
            is_supply_center=True,
            home_power=Power('Austria'),
            province_type=ProvinceType.LAND,
        ),
        'BUD': Province(
            name='BUD',
            neighbors={'VIE', 'GAL', 'RUM', 'SER', 'TRI'},
            is_supply_center=True,
            home_power=Power('Austria'),
            province_type=ProvinceType.LAND,
        ),
        'TRI': Province(
            name='TRI',
            neighbors={'VEN', 'TYR', 'VIE', 'BUD', 'SER', 'ADR', 'ALB'},
            is_supply_center=True,
            home_power=Power('Austria'),
            province_type=ProvinceType.COAST,
        ),
        'RUM': Province(
            name='RUM',
            neighbors={'SER', 'BUL', 'BLA', 'SEV', 'GAL', 'BUD', 'UKR'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'SER': Province(
            name='SER',
            neighbors={'BUD', 'RUM', 'BUL', 'GRE', 'ALB', 'TRI'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'BUL': Province(
            name='BUL',
            neighbors={'RUM', 'GRE', 'SER', 'CON', 'AEG', 'BLA'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'CON': Province(
            name='CON',
            neighbors={'BUL', 'ANK', 'SMY', 'AEG', 'BLA'},
            is_supply_center=True,
            home_power=Power('Turkey'),
            province_type=ProvinceType.COAST,
        ),
        'ANK': Province(
            name='ANK',
            neighbors={'CON', 'SMY', 'ARM', 'BLA'},
            is_supply_center=True,
            home_power=Power('Turkey'),
            province_type=ProvinceType.COAST,
        ),
        'SMY': Province(
            name='SMY',
            neighbors={'CON', 'ANK', 'ARM', 'SYR', 'AEG', 'EAS'},
            is_supply_center=True,
            home_power=Power('Turkey'),
            province_type=ProvinceType.COAST,
        ),
        'ARM': Province(
            name='ARM',
            neighbors={'SEV', 'BLA', 'ANK', 'SMY', 'SYR'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'SYR': Province(
            name='SYR',
            neighbors={'SMY', 'ARM', 'EAS'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'ALB': Province(
            name='ALB',
            neighbors={'TRI', 'SER', 'ADR', 'GRE', 'ION'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'GRE': Province(
            name='GRE',
            neighbors={'ALB', 'SER', 'BUL', 'AEG', 'ION'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'AEG': Province(
            name='AEG',
            neighbors={'ION', 'GRE', 'BUL', 'CON', 'SMY', 'EAS'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'ION': Province(
            name='ION',
            neighbors={'ADR', 'ALB', 'GRE', 'AEG', 'EAS', 'TUN', 'NAP', 'TYS', 'APU'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'EAS': Province(
            name='EAS',
            neighbors={'AEG', 'ION', 'SMY', 'SYR'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'ADR': Province(
            name='ADR',
            neighbors={'VEN', 'TRI', 'ALB', 'ION', 'APU'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'VEN': Province(
            name='VEN',
            neighbors={'TYR', 'TRI', 'ADR', 'APU', 'ROM', 'PIE', 'TUS'},
            is_supply_center=True,
            home_power=Power('Italy'),
            province_type=ProvinceType.COAST,
        ),
        'ROM': Province(
            name='ROM',
            neighbors={'VEN', 'TUS', 'NAP', 'APU', 'TYS'},
            is_supply_center=True,
            home_power=Power('Italy'),
            province_type=ProvinceType.COAST,
        ),
        'NAP': Province(
            name='NAP',
            neighbors={'ROM', 'APU', 'ION', 'TYS'},
            is_supply_center=True,
            home_power=Power('Italy'),
            province_type=ProvinceType.COAST,
        ),
        'TUS': Province(
            name='TUS',
            neighbors={'PIE', 'VEN', 'ROM', 'TYS', 'TYR'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'PIE': Province(
            name='PIE',
            neighbors={'MAR', 'TUS', 'VEN', 'TYR', 'LYO'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'APU': Province(
            name='APU',
            neighbors={'VEN', 'ADR', 'ION', 'NAP', 'ROM'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'TUN': Province(
            name='TUN',
            neighbors={'NAF', 'ION', 'TYS', 'WES'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'NAF': Province(
            name='NAF',
            neighbors={'TUN', 'WES', 'MAO'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'WES': Province(
            name='WES',
            neighbors={'LYO', 'TYS', 'TUN', 'NAF', 'SPA', 'MAO'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'TYS': Province(
            name='TYS',
            neighbors={'ROM', 'NAP', 'ION', 'TUN', 'WES', 'LYO', 'TUS'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'LYO': Province(
            name='LYO',
            neighbors={'WES', 'TYS', 'PIE', 'MAR', 'SPA', 'TUS'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.SEA,
        ),
        'MAR': Province(
            name='MAR',
            neighbors={'SPA', 'PIE', 'LYO', 'BUR', 'GAS'},
            is_supply_center=True,
            home_power=Power('France'),
            province_type=ProvinceType.COAST,
        ),
        'SPA': Province(
            name='SPA',
            neighbors={'POR', 'GAS', 'MAR', 'LYO', 'WES', 'MAO'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'POR': Province(
            name='POR',
            neighbors={'SPA', 'MAO'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'GAS': Province(
            name='GAS',
            neighbors={'SPA', 'MAR', 'BUR', 'PAR', 'BRE', 'MAO'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'BUR': Province(
            name='BUR',
            neighbors={'MAR', 'GAS', 'PAR', 'PIC', 'BEL', 'RUH', 'MUN'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'PAR': Province(
            name='PAR',
            neighbors={'GAS', 'BUR', 'PIC', 'BRE'},
            is_supply_center=True,
            home_power=Power('France'),
            province_type=ProvinceType.LAND,
        ),
        'BRE': Province(
            name='BRE',
            neighbors={'PIC', 'PAR', 'GAS', 'MAO', 'ENG'},
            is_supply_center=True,
            home_power=Power('France'),
            province_type=ProvinceType.COAST,
        ),
        'PIC': Province(
            name='PIC',
            neighbors={'BEL', 'BUR', 'PAR', 'BRE', 'ENG'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'BEL': Province(
            name='BEL',
            neighbors={'HOL', 'RUH', 'BUR', 'PIC', 'NTH', 'ENG'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        'RUH': Province(
            name='RUH',
            neighbors={'BUR', 'BEL', 'HOL', 'KIE', 'MUN'},
            is_supply_center=False,
            home_power=None,
            province_type=ProvinceType.LAND,
        ),
        'HOL': Province(
            name='HOL',
            neighbors={'BEL', 'RUH', 'KIE', 'NTH', 'HEL'},
            is_supply_center=True,
            home_power=None,
            province_type=ProvinceType.COAST,
        ),
        }
    return board




def standard_initial_state() -> GameState:
    board = standard_board()
    units: Dict[str, Unit] = {
        "EDI": Unit(Power("England"), "EDI", UnitType.FLEET),
        "LON": Unit(Power("England"), "LON", UnitType.FLEET),
        "LVP": Unit(Power("England"), "LVP", UnitType.ARMY),
        "STP": Unit(Power("Russia"), "STP", UnitType.FLEET),
        "MOS": Unit(Power("Russia"), "MOS", UnitType.ARMY),
        "WAR": Unit(Power("Russia"), "WAR", UnitType.ARMY),
        "SEV": Unit(Power("Russia"), "SEV", UnitType.FLEET),
        "KIE": Unit(Power("Germany"), "KIE", UnitType.FLEET),
        "MUN": Unit(Power("Germany"), "MUN", UnitType.ARMY),
        "BER": Unit(Power("Germany"), "BER", UnitType.ARMY),
        "VIE": Unit(Power("Austria"), "VIE", UnitType.ARMY),
        "BUD": Unit(Power("Austria"), "BUD", UnitType.ARMY),
        "TRI": Unit(Power("Austria"), "TRI", UnitType.FLEET),
        "ANK": Unit(Power("Turkey"), "ANK", UnitType.FLEET),
        "CON": Unit(Power("Turkey"), "CON", UnitType.ARMY),
        "SMY": Unit(Power("Turkey"), "SMY", UnitType.ARMY),
        "ROM": Unit(Power("Italy"), "ROM", UnitType.ARMY),
        "VEN": Unit(Power("Italy"), "VEN", UnitType.ARMY),
        "NAP": Unit(Power("Italy"), "NAP", UnitType.FLEET),
        "PAR": Unit(Power("France"), "PAR", UnitType.ARMY),
        "MAR": Unit(Power("France"), "MAR", UnitType.ARMY),
        "BRE": Unit(Power("France"), "BRE", UnitType.FLEET),
    }
    powers: Set[Power] = {
        Power("England"),
        Power("Russia"),
        Power("Germany"),
        Power("Austria"),
        Power("Turkey"),
        Power("Italy"),
        Power("France"),
    }
    return GameState(board=board, units=units, powers=powers)


__all__ = [
    "build_graph",
    "square_board",
    "triangle_board",
    "cooperative_attack_initial_state",
    "mesh_board_5x3",
    "demo_state_mesh",
    "fleet_coast_demo_state",
    "standard_board",
    "standard_initial_state",
]
