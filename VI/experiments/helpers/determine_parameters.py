def number_parameters(M, B):
    return M ** 2 + M + M * B + B


def number_parameters_complete(M, B, dim_x):
    return number_parameters(M, B) + M * dim_x



def par_in_range(par, range):
    min, max = range
    return (par < max) & (par > min)


def get_all_settings_in_range(range_):
    list_of_settings = []
    for states in range(100):
        for bases in range(21):
            if par_in_range(number_parameters(states, bases), range_):
                if states > 5:
                    if bases < 21:
                        list_of_settings.append((states, bases))
                    # print(states, bases, number_parameters(states, bases))
    return list_of_settings


def get_range(par, acceptance):
    return par - acceptance, par + acceptance


def number_parameters_sindy(input_dim):
    interaction_terms = input_dim * (input_dim - 1)/2  # e.g. x*y
    self_interaction = input_dim  # e.g. x^x
    no_interaction = input_dim  # e.g. x
    bias_term = input_dim  # scalar for each dimension
    return input_dim * (interaction_terms + self_interaction + no_interaction) + bias_term


if __name__ == '__main__':
    # for par in [1000]:
    par = 2000
    acceptance = 50
    list_of_state_base_pairs = get_all_settings_in_range(get_range(par, acceptance))
    # print(sorted(list_of_state_base_pairs, key=lambda x: x[1]))



    # print(number_parameters(22, 20))
    # print(number_parameters(42, 50))
    # print(number_parameters(26, 50))
    # print(number_parameters(12, 5))

    # print(number_parameters_complete(22, 20, 3))
    # print(number_parameters_complete(42, 50, 10))
    # print(number_parameters_complete(26, 50, 3))
    # print(number_parameters_complete(12, 5, 50))

    print(number_parameters_sindy(3))
    print(number_parameters_sindy(10))
    print(number_parameters_sindy(50))
