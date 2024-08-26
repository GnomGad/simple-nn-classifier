const fs = require("fs");

function loadTrainingData(fileName = "training-data.json") {
    const rawData = fs.readFileSync(fileName, "utf8");
    return JSON.parse(rawData);
}

function loadModel(fileName) {
    const rawData = fs.readFileSync(fileName, "utf8");
    return JSON.parse(rawData);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
    return x * (1 - x);
}

function applyThreshold(value) {
    return value >= 0.5 ? 1 : 0;
}

module.exports = {
    loadTrainingData,
    loadModel,
    sigmoid,
    sigmoidDerivative,
    applyThreshold,
};
