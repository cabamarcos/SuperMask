def save_params(individuo, filename = "individuo.txt"):
    with open(filename, "w") as f:
        #I want to see all params
        for name, param in individuo.named_parameters():
            f.write(f"{name} {param}\n")