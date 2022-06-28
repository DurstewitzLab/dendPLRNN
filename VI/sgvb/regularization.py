import torch as tc


def set_alphas(reg_ratios, reg_alphas, n_states):
    reg_ratios = tc.tensor(reg_ratios)
    reg_alphas = list(reg_alphas)
    reg_alphas.append(0.)
    reg_alphas = tc.tensor(reg_alphas)
    reg_group_n_states = distribute_states_by_ratios(n_states=n_states, reg_ratios=reg_ratios)
    alphas = tc.cat([a * tc.ones(d) for a, d in zip(reg_alphas, reg_group_n_states)])
    return alphas


def prepare_ratios(ratios):
    assert ratios.sum() <= 1
    missing_part = tc.abs(1 - ratios.sum())
    ratio_list = list(ratios)
    ratio_list.append(missing_part)
    return tc.tensor(ratio_list)


def distribute_states_by_ratios(n_states, reg_ratios):
    reg_ratios = prepare_ratios(reg_ratios)
    numbers_states = tc.round(n_states * reg_ratios.float())
    difference = n_states - numbers_states.sum()
    biggest_diff_at = tc.argmax(tc.abs(n_states * reg_ratios - numbers_states))
    numbers_states[biggest_diff_at] += difference
    numbers_states = numbers_states.int()
    return numbers_states


class Regularizer:
    def __init__(self, args):
        self.alphas = set_alphas(reg_ratios=args.reg_ratios, reg_alphas=args.reg_alphas, n_states=args.dim_z)

    def loss_regularized_parameter(self, parameter, to_value, weighting_of_states):
        diff = parameter - to_value * tc.ones(len(parameter))
        loss = tc.sum(weighting_of_states * tc.pow(diff, 2))
        return loss

    def loss(self, parameters):
        A, W, h = parameters
        loss = 0.
        loss += self.loss_regularized_parameter(parameter=A, to_value=1., weighting_of_states=self.alphas)
        loss += self.loss_regularized_parameter(parameter=W, to_value=0., weighting_of_states=self.alphas)
        loss += self.loss_regularized_parameter(parameter=h, to_value=0., weighting_of_states=self.alphas)
        return loss
