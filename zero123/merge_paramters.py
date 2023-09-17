import torch


zero123_xl_missing_keys = ['cond_stage_model.transformer.text_model.embeddings.position_ids', 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight', 'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.bias', 'cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.weight', 'cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.bias', 'cond_stage_model.transformer.text_model.final_layer_norm.weight', 'cond_stage_model.transformer.text_model.final_layer_norm.bias']

path1 = "zero123-xl.ckpt"
path2 = "control_sd15_canny.pth"


zero_123 = torch.load(path1)
control =torch.load(path2)

zero_state= zero_123["state_dict"]
control_state = control["state_dict"]


has_keys = 0
for item in zero123_xl_missing_keys:
    if item in control_state:
        has_keys += 1
print(has_keys, len(zero123_xl_missing_keys))
