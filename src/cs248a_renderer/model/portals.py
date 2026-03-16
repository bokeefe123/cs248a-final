from dataclasses import dataclass, field
from pyglm import glm
import slangpy as spy
from typing import List, Dict

from cs248a_renderer.model.scene_object import SceneObject


@dataclass
class Portal(SceneObject):
    """A rectangular portal that teleports rays to its partner portal."""

    vertices: List[glm.vec3] = field(
        default_factory=lambda: [
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 0.0, 0.0),
        ]
    )
    partner_name: str = ""
    glow_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.3, 0.5, 1.0))
    glow_width: float = 0.05
    glow_intensity: float = 3.0
    _partner_index: int = 0

    def __post_init__(self):
        self.update_param(self.get_vertices())

    def update_param(self, vertices: List[glm.vec3]):
        self.bottomLeftVertex = vertices[0]
        self.bottomEdge = vertices[1] - vertices[0]
        self.leftEdge = vertices[3] - vertices[0]
        cross_product = glm.cross(self.bottomEdge, self.leftEdge)
        self.normal = glm.normalize(cross_product)

    def get_vertices(self) -> List[glm.vec3]:
        trans_mat = self.get_transform_matrix()
        return [glm.vec3(trans_mat @ glm.vec4(vertex, 1.0)) for vertex in self.vertices]

    def get_this(self) -> Dict:
        self.update_param(self.get_vertices())
        return {
            "bottomLeftVertex": self.bottomLeftVertex.to_list(),
            "bottomEdge": self.bottomEdge.to_list(),
            "leftEdge": self.leftEdge.to_list(),
            "normal": self.normal.to_list(),
            "partnerIndex": self._partner_index,
            "glowColor": self.glow_color.to_list(),
            "glowWidth": self.glow_width,
            "glowIntensity": self.glow_intensity,
        }


def resolve_partner_indices(portals: List[Portal]) -> None:
    """Resolve partner_name references to buffer indices."""
    name_to_index = {p.name: i for i, p in enumerate(portals)}
    for portal in portals:
        if portal.partner_name in name_to_index:
            portal._partner_index = name_to_index[portal.partner_name]


def create_portal_buf(
    module: spy.Module, portals: List[Portal]
) -> spy.NDBuffer:
    device = module.device
    buffer = spy.NDBuffer(
        device=device,
        dtype=module.Portal.as_struct(),
        shape=(max(len(portals), 1),),
    )
    cursor = buffer.cursor()
    for idx, portal in enumerate(portals):
        cursor[idx].write(portal.get_this())
    cursor.apply()
    return buffer
