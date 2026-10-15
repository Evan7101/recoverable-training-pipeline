import json, pathlib
def train(steps=10):
    state={'step':0,'weight':0.0}
    for _ in range(steps):
        state['weight']-=0.1*(2*(state['weight']-3)); state['step']+=1
    pathlib.Path('checkpoint.json').write_text(json.dumps(state))
    return state
if __name__ == '__main__': print(train())
