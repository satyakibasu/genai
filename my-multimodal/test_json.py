import json
import pandas as pd

with open("contract_response.json",'r') as file:
    lines = file.readlines() # gives a list object
    new_lines = lines[1:-1] # remove the 1st and last line

'''
# rewrite into a new json file
with open("a.json",'w') as f:
    f.writelines(new_lines)
'''

with open("a.json",'r') as t:
    data = json.load(t)

data = json.load(new_lines)

trader = data['trader']
trades = data['trades']
print(type(trader))

trader_df = pd.DataFrame(trader,index=[0])
trades_df = pd.DataFrame(trades)

print(trader_df)
print(trades_df)

    