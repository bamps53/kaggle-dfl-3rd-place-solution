def rescale_layer_norm(model, state_dict):
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        new_shape = model_state_dict[k].shape
        old_shape = v.shape
        if new_shape != old_shape:
            print(f'rescale {k} from {old_shape} -> {new_shape}')
            state_dict[k] = torch.nn.functional.interpolate(
                v[None, None], new_shape, mode='bilinear').squeeze()
    return state_dict
