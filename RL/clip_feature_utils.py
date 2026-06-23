import torch


def as_feature_tensor(features):
    """Return projected CLIP features for both old and new transformers APIs."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if isinstance(features, (tuple, list)):
        for item in features:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item
    raise TypeError(f"Cannot extract CLIP feature tensor from {type(features)!r}")
