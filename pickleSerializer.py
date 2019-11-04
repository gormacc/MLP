import pickle

def serialize(filePath, objectToSerialize):
    pickleOut = open(filePath,"wb")
    pickle.dump(objectToSerialize, pickleOut)
    pickleOut.close()

def deserialize(filePath):
    pickleIn = open(filePath,"rb")
    deserializedObject = pickle.load(pickleIn)
    pickleIn.close()
    return deserializedObject

