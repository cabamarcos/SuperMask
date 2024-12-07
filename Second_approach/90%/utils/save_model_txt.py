def save_params(individuo, filename="individuo.txt"):
    with open(filename, "w") as f:
        state_dict = individuo.state_dict()
        for name, param in state_dict.items():
            f.write(f"{name} {param}\n")