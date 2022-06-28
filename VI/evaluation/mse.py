import torch as tc


def get_input(inputs, step, n_steps):
    if inputs is not None:
        input_for_step = inputs[step:(-n_steps + step), :]
    else:
        input_for_step = None
    return input_for_step


def get_ahead_pred_obs(model, data, n_steps, inputs):
    time_steps = len(data) - n_steps
    x_data = data[:-n_steps, :]
    z, _ = model.rec_model(x_data)
    x_pred = []
    for step in range(n_steps):
        z = model.gen_model.latent_step(z, get_input(inputs, step, n_steps))
        x_pred.append(model.gen_model.observation(z))
    x_pred = tc.cat(x_pred)
    x_pred = tc.reshape(x_pred, shape=(n_steps, time_steps, -1))
    return x_pred


def construct_ground_truth(data, n_steps):
    time_steps = len(data) - n_steps
    x_true = [data[step:, :] for step in range(1, n_steps + 1)]
    x_true = tc.stack([x[:time_steps, :] for x in x_true])
    return x_true


def squared_error(x_pred, x_true):
    return tc.pow(x_pred - x_true, 2)


def n_steps_ahead_pred_mse(model, data, n_steps, inputs=None):
    with tc.no_grad():
        x_pred = get_ahead_pred_obs(model, data, n_steps, inputs)
        x_true = construct_ground_truth(data, n_steps)
        mean_squared_error = squared_error(x_pred, x_true).mean([1, 2]).numpy()
    return mean_squared_error
