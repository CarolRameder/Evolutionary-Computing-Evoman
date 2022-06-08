import pandas as pd
import ast

data = []
with open("final2.txt", "r") as inFile:
    data = ast.literal_eval(inFile.read())

df = pd.DataFrame(data)

def enemies_beaten(fights):
    enemies_beaten = 0
    for fight in fights:
        if fight["enemy_life_avg"] == 0:
           enemies_beaten = enemies_beaten + 1
    return enemies_beaten

print(df.apply(lambda x: enemies_beaten(x['fights']),axis=1))

print(df.loc[[34]])
