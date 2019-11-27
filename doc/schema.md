LevelDB keys and values.

General principles: Keys should be all UTF-8 strings, but values can be binary.

key: "sourceFrame:%08x", the parameter is the hex frame number. Values are a SourceFrame flatbuffer.

key: "sourceImage:%016x", the parameter is a hex hash of the soureframe RGB planes. Values are the three color planes.

