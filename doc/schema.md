LevelDB keys and values.

General principles: Keys should be all UTF-8 strings, but values can be binary.

key: "sourceFrame:%08x", the parameter is the hex frame number. Values are a SourceFrame flatbuffer.

key: "sourceImage:%016x", the parameter is a hex hash of the soureframe RGB planes. Values are the three color planes.

key: "quantizeMap:%016x", the parameter is the hex hash of the source image, the value string "%016x" hex of the
    quantized image hash.

key: "quantImage:%016x", the parameter is the hex hash of the quant image, the value kFrameSizeBytes of color indices.

