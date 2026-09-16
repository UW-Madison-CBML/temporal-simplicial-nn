"""TGAT baseline."""

from TGNN_models.dyglib_runner import DyGLibModelConfig, run_dyglib_model


def run(bundle, config, device):
  dyglib_config = DyGLibModelConfig(
      lr=config.lr,
      epochs=config.epochs,
      memory_dim=config.memory_dim,
      time_dim=config.time_dim,
      embedding_dim=config.embedding_dim,
      num_neighbors=config.num_neighbors,
      seed=config.seed,
      output_dim=config.embedding_dim,
      time_feat_dim=config.time_dim,
      max_batches=getattr(config, "max_batches", None),
  )
  return run_dyglib_model("tgat", bundle, dyglib_config, device)
