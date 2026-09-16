# Third-party code

This repository vendors two external projects. Both are MIT licensed, and their
license texts are included alongside the code.

| Path | Upstream | License |
|---|---|---|
| `model/topomodelx/` | TopoModelX (pyt-team) | MIT — `model/topomodelx/LICENSE` |
| `baselines/dyglib/` | DyGLib | MIT — `baselines/dyglib/LICENSE` |

`model/topomodelx/` supplies the simplicial message-passing layers the proposed
model builds on; only the simplicial subset is vendored.

`baselines/dyglib/` supplies the temporal-graph backbones used by the baselines
(TGAT, JODIE, GraphMixer, DyGFormer).

Modifications made for this work are described in the README; upstream files are
otherwise unchanged.
