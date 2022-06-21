import torch as tc


def l1_norm(x):
    return tc.abs(x)


def l2_norm(x):
    return tc.pow(x, 2)


def set_norm(reg_norm):
    norm = l2_norm
    if reg_norm == 'l1':
        norm = l1_norm
    return norm


def set_alphas(reg_ratios, reg_alphas, n_states_total):
    reg_group_ratios = tc.tensor(reg_ratios)
    reg_group_alphas = reg_alphas
    reg_group_alphas = list(reg_group_alphas)
    reg_group_alphas.append(0.)
    reg_group_alphas = tc.tensor(reg_group_alphas)
    reg_group_n_states = distribute_states_by_ratios(n_states=n_states_total, ratios_states=reg_group_ratios)
    alphas = tc.cat([a * tc.ones(d) for a, d in zip(reg_group_alphas, reg_group_n_states)])
    # regularize non-read-out states
    return alphas.flip((0,))


def prepare_ratios(ratios):
    assert ratios.sum() <= 1
    missing_part = tc.abs(1 - ratios.sum())
    ratio_list = list(ratios)
    ratio_list.append(missing_part)
    return tc.tensor(ratio_list)


def distribute_states_by_ratios(n_states, ratios_states):
    assert n_states != 0
    ratios_states = prepare_ratios(ratios_states)
    numbers_states = tc.round(n_states * ratios_states.float())
    difference = n_states - numbers_states.sum()
    biggest_diff_at = tc.argmax(tc.abs(n_states * ratios_states - numbers_states))
    numbers_states[biggest_diff_at] += difference
    numbers_states = numbers_states.int()
    return numbers_states


class Regularizer:
    def __init__(self, args):
        self.norm = set_norm(reg_norm=args.reg_norm)
        self.alphas = set_alphas(reg_ratios=args.reg_ratios, reg_alphas=args.reg_alphas,
                                 n_states_total=args.dim_z)

    def loss_regularized_parameter(self, parameter, to_value, weighting_of_states):
        diff = parameter - to_value * tc.ones(len(parameter), device=parameter.device)
        loss = tc.sum(weighting_of_states * self.norm(diff))
        return loss

    def loss(self, parameters):
        A, W, h = parameters
        loss = 0.
        loss += self.loss_regularized_parameter(parameter=A, to_value=1., weighting_of_states=self.alphas)
        loss += self.loss_regularized_parameter(parameter=W, to_value=0., weighting_of_states=self.alphas.view(-1 , 1))
        loss += self.loss_regularized_parameter(parameter=h, to_value=0., weighting_of_states=self.alphas)
        return loss
    
    def to(self, device: tc.device) -> None:
        self.alphas = self.alphas.to(device)