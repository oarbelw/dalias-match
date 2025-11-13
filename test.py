import pickle, os
with open("/Users/orenarbel-wood/Desktop/Dovie/artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)
print(type(artifacts), list(artifacts.keys())[:5])
print(os.path.getsize("/Users/orenarbel-wood/Desktop/Dovie/artifacts.pkl")/1e6, "MB")
