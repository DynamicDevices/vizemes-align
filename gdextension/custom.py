# Loaded by godot-cpp/SConstruct (customs). Quiets known godot-cpp header noise.


def configure(env):
    env.Append(CCFLAGS=["-Wno-unused-parameter"])
    env.Append(CXXFLAGS=["-Wno-unused-parameter", "-Wno-unused-variable"])
