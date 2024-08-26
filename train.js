const fs = require("fs");
const { loadTrainingData, sigmoid, sigmoidDerivative, loadModel } = require("./lib");

function createModel() {
    return {
        hiddenWeights1: [Math.random(), Math.random()],
        hiddenWeights2: [Math.random(), Math.random()],
        hiddenBias1: Math.random(),
        hiddenBias2: Math.random(),
        outputWeights: [Math.random(), Math.random()],
        outputBias: Math.random(),
    };
}

function saveModel(fileName, model) {
    fs.writeFileSync(fileName, JSON.stringify(model), "utf8");
}

function forwardPropagation(inputs, model) {
    let hiddenOutput1 = sigmoid(
        inputs[0] * model.hiddenWeights1[0] + inputs[1] * model.hiddenWeights1[1] + model.hiddenBias1
    );
    let hiddenOutput2 = sigmoid(
        inputs[0] * model.hiddenWeights2[0] + inputs[1] * model.hiddenWeights2[1] + model.hiddenBias2
    );

    let output = sigmoid(
        hiddenOutput1 * model.outputWeights[0] + hiddenOutput2 * model.outputWeights[1] + model.outputBias
    );
    return output;
}

function train(model, trainingData, iterations, learningRate) {
    for (let i = 0; i < iterations; i++) {
        trainingData.forEach((data) => {
            let hiddenOutput1 = sigmoid(
                data.input[0] * model.hiddenWeights1[0] + data.input[1] * model.hiddenWeights1[1] + model.hiddenBias1
            );
            let hiddenOutput2 = sigmoid(
                data.input[0] * model.hiddenWeights2[0] + data.input[1] * model.hiddenWeights2[1] + model.hiddenBias2
            );

            let output = sigmoid(
                hiddenOutput1 * model.outputWeights[0] + hiddenOutput2 * model.outputWeights[1] + model.outputBias
            );

            let error = data.output - output;

            let deltaOutput = error * sigmoidDerivative(output);
            model.outputWeights[0] += deltaOutput * hiddenOutput1 * learningRate;
            model.outputWeights[1] += deltaOutput * hiddenOutput2 * learningRate;
            model.outputBias += deltaOutput * learningRate;

            let deltaHidden1 = deltaOutput * model.outputWeights[0] * sigmoidDerivative(hiddenOutput1);
            let deltaHidden2 = deltaOutput * model.outputWeights[1] * sigmoidDerivative(hiddenOutput2);

            model.hiddenWeights1[0] += deltaHidden1 * data.input[0] * learningRate;
            model.hiddenWeights1[1] += deltaHidden1 * data.input[1] * learningRate;
            model.hiddenBias1 += deltaHidden1 * learningRate;

            model.hiddenWeights2[0] += deltaHidden2 * data.input[0] * learningRate;
            model.hiddenWeights2[1] += deltaHidden2 * data.input[1] * learningRate;
            model.hiddenBias2 += deltaHidden2 * learningRate;
        });
    }
}

const trainingData = loadTrainingData("training-data.json");

let xorModel = createModel();
train(xorModel, trainingData.XOR, 100000, 0.1);
saveModel("model_xor.json", xorModel);

console.log("Predictions for XOR:");
trainingData.XOR.forEach((data) => {
    let output = forwardPropagation(data.input, xorModel);
    console.log(`Input: ${data.input}, Expected: ${data.output}, Predicted: ${output.toFixed(4)}`);
});

let andModel = createModel();
train(andModel, trainingData.AND, 100000, 0.1);
saveModel("model_and.json", andModel);

console.log("Predictions for AND:");
trainingData.AND.forEach((data) => {
    let output = forwardPropagation(data.input, andModel);
    console.log(`Input: ${data.input}, Expected: ${data.output}, Predicted: ${output.toFixed(4)}`);
});
