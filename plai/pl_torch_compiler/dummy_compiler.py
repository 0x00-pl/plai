def custom_compiler(gm, example_inputs):
    print("Using custom dummy compiler!")
    return gm.forward
