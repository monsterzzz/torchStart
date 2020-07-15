# import pandas


# df = pandas.DataFrame([["a",2,"c"], ["b",4,"a"], ["c",7,"b"], ["d",4,"d"]])
# df.columns = ["col1","col2","col3"]
# df.index = ["i1","i2","i3","i4"]

# col1 = pandas.get_dummies(df["col1"],drop_first=True)
# col3 = pandas.get_dummies(df["col3"],drop_first=True)
# print(df)

# df = pandas.concat([df,col1],axis=1)
# df.drop(["col1"],axis=1,inplace=True)

# df = pandas.concat([df,col3],axis=1)
# df.drop(["col3"],axis=1,inplace=True)

# print(df)

a = [1,2,3,4,5,7]
print(a[-5:])