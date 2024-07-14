def custom_compiler(gm, example_inputs):
    print("Using custom compiler!")
    return gm.forward
