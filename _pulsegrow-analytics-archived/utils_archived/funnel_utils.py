def calculate_funnel(data, steps):
    funnel = {}
    total_users = len(data)
    for step in steps:
        users_in_step = len(data[data['funnel_step'] == step])
        funnel[step] = users_in_step / total_users
    return funnel