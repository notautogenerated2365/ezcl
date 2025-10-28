const numMeta = {
    INT8: {className: "int8", numName: "char"},
    INT16: {className: "int16", numName: "short"},
    INT32: {className: "int32", numName: "int"},
    INT64: {className: "int64", numName: "long long int"},
    UINT8: {className: "uint8", numName: "unsigned char"},
    UINT16: {className: "uint16", numName: "unsigned short"},
    UINT32: {className: "uint32", numName: "unsigned int"},
    UINT64: {className: "uint64", numName: "unsigned long long int"},
    FLOAT16: {className: "float16", numName: "not yet implemented"},
    FLOAT32: {className: "float32", numName: "float"},
    FLOAT64: {className: "float64", numName: "double"},
};

const opMeta = {
    ADD: {name: "add", capsName: "ADD", op: "+"},
    SUB: {name: "sub", capsName: "SUB", op: "-"},
    MUL: {name: "mul", capsName: "MUL", op: "*"},
    DIV: {name: "div", capsName: "DIV", op: "/"},
};

module.exports = {
    numMeta,
    opMeta,
};