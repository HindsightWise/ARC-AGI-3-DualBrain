"""
Suite of 25 Novel ARC-AGI-3 Game Environments across 5 Paradigms.
"""
from typing import Dict, Type
from arcengine import ARCBaseGame

from scripts.novel_games.p1_navigation import (
    NG01_WormholeTunnel,
    NG02_MovingHazard,
    NG03_LightShadow,
    NG04_GravityWells,
    NG05_CollapsingPath,
)
from scripts.novel_games.p2_physics import (
    NG06_IceSlider,
    NG07_BoulderBridge,
    NG08_MagnetPolarity,
    NG09_HeavyLightBoxes,
    NG10_PortalBoxRouting,
)
from scripts.novel_games.p3_logic import (
    NG11_LogicCircuit,
    NG12_ColorMixer,
    NG13_SymmetryPainter,
    NG14_SudokuLight,
    NG15_SequenceMelody,
)
from scripts.novel_games.p4_multiagent import (
    NG16_QuadMirrors,
    NG17_PredatorPrey,
    NG18_ConvoyEscort,
    NG19_ShadowClone,
    NG20_SplitMerge,
)
from scripts.novel_games.p5_affordance import (
    NG21_TumblerLock,
    NG22_PipeFlow,
    NG23_SlidingTile8,
    NG24_OpticalBeamSplit,
    NG25_ResourceCrafting,
)

ALL_NOVEL_GAMES: Dict[str, Type[ARCBaseGame]] = {
    # Paradigm 1: Spatial & Topological Navigation
    "ng01": NG01_WormholeTunnel,
    "ng02": NG02_MovingHazard,
    "ng03": NG03_LightShadow,
    "ng04": NG04_GravityWells,
    "ng05": NG05_CollapsingPath,
    # Paradigm 2: Push-Box & Object Physics
    "ng06": NG06_IceSlider,
    "ng07": NG07_BoulderBridge,
    "ng08": NG08_MagnetPolarity,
    "ng09": NG09_HeavyLightBoxes,
    "ng10": NG10_PortalBoxRouting,
    # Paradigm 3: Pattern Matching & Logic Gates
    "ng11": NG11_LogicCircuit,
    "ng12": NG12_ColorMixer,
    "ng13": NG13_SymmetryPainter,
    "ng14": NG14_SudokuLight,
    "ng15": NG15_SequenceMelody,
    # Paradigm 4: Multi-Agent & Desynchronization
    "ng16": NG16_QuadMirrors,
    "ng17": NG17_PredatorPrey,
    "ng18": NG18_ConvoyEscort,
    "ng19": NG19_ShadowClone,
    "ng20": NG20_SplitMerge,
    # Paradigm 5: Affordance & Point-and-Click Mechanisms
    "ng21": NG21_TumblerLock,
    "ng22": NG22_PipeFlow,
    "ng23": NG23_SlidingTile8,
    "ng24": NG24_OpticalBeamSplit,
    "ng25": NG25_ResourceCrafting,
}

__all__ = [
    "ALL_NOVEL_GAMES",
    "NG01_WormholeTunnel",
    "NG02_MovingHazard",
    "NG03_LightShadow",
    "NG04_GravityWells",
    "NG05_CollapsingPath",
    "NG06_IceSlider",
    "NG07_BoulderBridge",
    "NG08_MagnetPolarity",
    "NG09_HeavyLightBoxes",
    "NG10_PortalBoxRouting",
    "NG11_LogicCircuit",
    "NG12_ColorMixer",
    "NG13_SymmetryPainter",
    "NG14_SudokuLight",
    "NG15_SequenceMelody",
    "NG16_QuadMirrors",
    "NG17_PredatorPrey",
    "NG18_ConvoyEscort",
    "NG19_ShadowClone",
    "NG20_SplitMerge",
    "NG21_TumblerLock",
    "NG22_PipeFlow",
    "NG23_SlidingTile8",
    "NG24_OpticalBeamSplit",
    "NG25_ResourceCrafting",
]
