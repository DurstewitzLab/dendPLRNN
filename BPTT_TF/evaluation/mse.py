import torch as tc
from torch.linalg import pinv


def get_input(inputs, step, n_steps):
    if inputs is not None:
        input_for_step = inputs[step:(-n_steps + step), :]
    else:
        input_for_step = None
    return input_for_step


def get_ahead_pred_obs(model, data, n_steps):
    # dims
    T, dx = data.size()

    # true data
    time_steps = T - n_steps
    x_data = data[:-n_steps, :].to(model.device)

    # latent model
    lat = model.latent_model

    # initial state. Uses x0 of the dataset
    # batch dim of z holds all T - n_steps time steps
    if model.z0_model:
        z = model.z0_model(x_data)
    else:
        dz = lat.d_z
        z = tc.randn((time_steps, dz), device=model.device)

        # obs. model inv?
        inv_obs = model.args['use_inv_tf']
        B_PI = None
        if inv_obs:
            B = model.output_layer.weight
            B_PI = pinv(B)
        z = lat.teacher_force(z, x_data, B_PI)

    X_pred = tc.empty((n_steps, time_steps, dx), device=model.device)
    params = model.get_latent_parameters()
    for step in range(n_steps):
        # latent step performs ahead prediction on every
        # time step here
        z = lat.latent_step(z, *params)
        x = model.output_layer(z)
        X_pred[step] = x

    return X_pred

def construct_ground_truth(data, n_steps):
    T, dx = data.size()
    time_steps = T - n_steps
    X_true = tc.empty((n_steps, time_steps, dx))
    for step in range(1, n_steps + 1):
        X_true[step - 1] = data[step : time_steps + step]
    return X_true


def squared_error(x_pred, x_true):
    return tc.pow(x_pred - x_true, 2)

@tc.no_grad()
def n_steps_ahead_pred_mse(model, data, n_steps):
    x_pred = get_ahead_pred_obs(model, data, n_steps)
    x_true = construct_ground_truth(data, n_steps).to(model.device)
    mse = squared_error(x_pred, x_true).mean([1, 2]).cpu().numpy()
    return mse
