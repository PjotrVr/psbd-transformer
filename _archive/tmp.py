class PsbdBlockWrapper(nn.Module):
    def __init__(self, base_block, dropout_rate):
        super().__init__()
        self.base_block = base_block
        self.post_addition_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_tensor):
        # We must replicate the internal forward pass of PyTorch's ViT block
        # to access the intermediate tensors directly after the residual additions.
        # This explicit disruption forces the network to expose over-indexed backdoor paths.

        normalized_input = self.base_block.ln_1(input_tensor)
        attention_output, _ = self.base_block.self_attention(
            normalized_input, normalized_input, normalized_input, need_weights=False
        )
        attention_output = self.base_block.dropout(attention_output)

        merged_attention = input_tensor + attention_output
        corrupted_attention = self.post_addition_dropout(merged_attention)

        normalized_corrupted = self.base_block.ln_2(corrupted_attention)
        mlp_output = self.base_block.mlp(normalized_corrupted)

        merged_mlp = corrupted_attention + mlp_output
        corrupted_output = self.post_addition_dropout(merged_mlp)

        return corrupted_output


def load_base_vision_transformer():
    return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)


def wrap_encoder_block(block, dropout_rate):
    return PsbdBlockWrapper(base_block=block, dropout_rate=dropout_rate)


def inject_psbd_dropout_layers(model, dropout_rate):
    # State mutation is isolated here to prevent side effects in the main execution flow.
    for index, block in enumerate(model.encoder.layers):
        model.encoder.layers[index] = wrap_encoder_block(block, dropout_rate)
    return model
